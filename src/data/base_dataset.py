import json
import functools
import torch
import base64
from PIL import Image
import io
import os
import numpy as np
import random
import logging
import webdataset as wds
import torch.distributed as dist
import ast
from scipy.optimize import linear_sum_assignment
from .base import image_augment, video_augment, gen_mixed_caption, read_frames_from_timestamps_ffmpeg, select_image_index_from_score, flip_scores, select_obelics_subsampled_text, select_mmc4_subsampled_text, read_frames_from_timestamps_and_path, obelics_optim_assignments, select_cc3m_subsampled_text


"""
since original webdataset have no __len__, we wrapper it to add __len__ attribute,
so that we can use it in pytorch dataloader + huggingface trainer
"""
class SizedWebDataset(wds.WebDataset):
    def __init__(self, *args, length, batch_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.length = length
        self.batch_size = batch_size
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def __len__(self):
        return self.length // (self.batch_size * self.world_size)

keys = [
    ('img_txt', 'iwt'),
    ('vid_txt', 'vwt'),
    ('inter_img_txt', 'inter_iwt'),
    ('inter_vid_txt', 'inter_vwt'),
]

class BaseDataset():
    def __init__(self, split, data_path, batch_size, tokenizer, image_processor=None, video_processor=None, dataset_params=None, custom_logger=None):
        self.split = split
        assert self.split in ["train", "val", "test"]
        self.custom_logger = custom_logger
        self.data_path = data_path
        self.data = []
        self.tokenizer = tokenizer
        assert dataset_params is not None
        self.dataset_params = dataset_params
        if 'fine_tuning' in self.dataset_params:
            self.fine_tuning = self.dataset_params['fine_tuning']
        else:
            self.fine_tuning = False
        if image_processor is None:
            self.image_processor = image_augment(mode=self.split)
        else:
            self.image_processor = self.select_image_augment(image_processor, mode=self.split)
        if video_processor is None:
            if self.fine_tuning:
                self.video_processor = video_augment(video_frame=self.dataset_params['vid_instruct']['VIDEO_FRAMES'], video_image_size=self.dataset_params['vid_instruct']['VIDEO_IMAGE_SIZE'], mode=self.split)
            else:
                self.video_processor = video_augment(video_frame=self.dataset_params['vid_txt']['VIDEO_FRAMES'], video_image_size=self.dataset_params['vid_txt']['VIDEO_IMAGE_SIZE'], mode=self.split)
        else:
            self.video_processor = video_processor
        self.batch_size = batch_size
        self.custom_logger.info("dataset params is: {}".format(self.dataset_params))
        self.split_data_by_node = self.dataset_params['split_data_by_node'] and self.split == "train" # only split data for train
        self.use_azfuse = self.dataset_params['use_azfuse']
        # insturction fine-tuning
        if self.fine_tuning:
            self.img_instruct_data_root = self.dataset_params['img_instruct']['DATA_ROOT']
            self.vid_instruct_data_root = self.dataset_params['vid_instruct']['DATA_ROOT']
        else:
            for dataset_key, attr_prefix in keys:
                setattr(self, f"{attr_prefix}_tar_pre_cache", self.dataset_params[dataset_key]['tar_pre_cache'])
                setattr(self, f"{attr_prefix}_pre_cache_ratio", self.dataset_params[dataset_key]['pre_cache_ratio'])
        self.init_preprocess()

    def init_preprocess(self):
        if self.fine_tuning:
            self.preprocess_instruction_fn = functools.partial(self.preprocess_instruction)
            self.preprocess_instruction_image_fn = functools.partial(self.preprocess_image)
        else:
            self.preprocess_image_fn = functools.partial(self.preprocess_image)
            self.preprocess_video_fn = functools.partial(self.preprocess_video)
            self.preprocess_image_text_fn = functools.partial(self.preprocess_text, clean_data_use_strategy=self.dataset_params['img_txt']['clean_data_use_strategy'])
            self.preprocess_video_text_fn = functools.partial(self.preprocess_text, clean_data_use_strategy=self.dataset_params['vid_txt']['clean_data_use_strategy'])
            self.preprocess_interleaved_fn = functools.partial(self.preprocess_interleaved, text_coherence=self.dataset_params['inter_img_txt']['interlevel_text_coherence'], clean_data_use_strategy=self.dataset_params['inter_img_txt']['clean_data_use_strategy'], balanced_sampling=self.dataset_params['inter_img_txt']['balanced_sampling'])
            self.preprocess_interleaved_video_fn = functools.partial(self.preprocess_interleaved_video)
            self.preprocess_interleaved_video_wds_fn = functools.partial(self.preprocess_interleaved_video_wds)
        # add special tokens
        self.media_token_id = self.tokenizer("<visual>", add_special_tokens=False)["input_ids"][-1] # 50266
        self.endofchunk_token_id = self.tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1] # 50265
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids("<s>") # 0
        # add instruction special tokens
        if self.fine_tuning:
            self.human_token_id = self.tokenizer("<human>", add_special_tokens=False)["input_ids"][-1]
            self.gpt_token_id = self.tokenizer("<gpt>", add_special_tokens=False)["input_ids"][-1]

    def load_data(self):
        raise NotImplementedError

    def log_and_continue(self, exn):
        """Call in an exception handler to ignore any exception, issue a warning, and continue."""
        logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
        return True

    def select_image_augment(self, image_processor, mode='train'):
        """
        this train only include random resized crop.
        image_processor is a list of image augment function, [train, val]
        """
        if mode == 'train':
            return image_processor[0]
        else:
            return image_processor[1]

    
    def preprocess_text(self, sample, padding_rule="longest", add_blank_at_begin=True, clean_data_use_strategy="noisy_only"):
        """
        padding to max lengeth since video_text dataset may load text with different length
        if no generated caption, use original caption as default
        the gen captions is xxx<EOC>xxx<EOC>xxx<EOC>, five xxx in general
        """
        self.tokenizer.padding_side = "right"
        blank = ""
        if add_blank_at_begin:
            if random.random() <= 0.5:
                blank += " "
        padding_sample = []
        for s in sample:
            if len(s.split('\t')) > 1:
                original_caption = s.split('\t')[0].strip()
                gen_captions = s.split('\t')[1].split("<EOC>")
                gen_captions = [gen_caption.strip() for gen_caption in gen_captions]
                modifed_caption = gen_mixed_caption(original_caption, gen_captions, clean_data_use_strategy=clean_data_use_strategy)
            else:
                modifed_caption = s
            padding_sample.append(modifed_caption)
        padding_sample = [
            (f"<s><visual>{blank + s.strip()}<|endofchunk|>{self.tokenizer.eos_token}") for s in padding_sample
        ]
        text = self.tokenizer(
            padding_sample,
            max_length=32,
            padding=padding_rule,
            truncation="only_first",
            return_tensors="pt",
        )
        labels = text["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        labels[labels == self.media_token_id] = -100
         # predicting the EOS token, which indicates the end of the sentence
        text["labels"] = labels
        return text["input_ids"], text["attention_mask"], text["labels"]

    def preprocess_image(self, sample):
        image = [self.image_processor(s).unsqueeze(0) for s in sample]
        image = torch.cat(image, dim=0)
        return image

    def preprocess_video(self, sample):
        """
        # visual should be 6D input: (batch_size, Image in same chunk, Time, num_channels, height, width)
        # sample: [B, 1, T, H, W, C] ? 
        # return: [B, F (Image in same chunk), T(Time), C, H, W]
        # self.custom_logger.info(sample[0][0].shape) # ([317, 336, 596, 3]), T, H, W, C
        """
        video = [self.video_processor(s[0].permute(3, 0, 1, 2)).permute(1, 0, 2, 3).unsqueeze(0) for s in sample]  
        # from t,h,W,C TO C,T,H,W then to T,C,H,W since video_augment need this format
        video = torch.cat(video, dim=0)
        return video.unsqueeze(1)


    def clean_incomplete_obelics(self, sentences, image_info):
        """
        remove the image not download or preprocess successfully
        For example, sentences: [Text, None, None] Image: [None, None, Image]
        Then we should remove the 2th elements since no base64 avaiable
        """
        assert len(sentences) == len(image_info)
        new_sentences, new_image_info = [], []
        for i in range(len(sentences)):
            if sentences[i] is not None:
                new_sentences.append(sentences[i])
            if image_info[i] is not None:
                new_image_info.append(image_info[i])
        return new_sentences, new_image_info


    def sample_sequence_from_document(self, text, num_selected_tokens=64, max_num_images=3, image_appeared_order=None, shift_region=20):
        """
        Text: a document with a lot of images and texts. For example. <s>[Text1]<eoc><visual>[Text2]<eoc><eos>
        The shift region is to add some texts before first image. But should not be too large, otherwise the learnable sequences will be too short.
        Remain question:
            1. If always begin with <visual> token, the prompt will be missed especially for the downstream tasks.
            2. There may some cases the "<visual>" log is longer than image numbers.
        """
        text = (
            text.replace(" <|endofchunk|>", "<|endofchunk|>")
            .replace("<visual> ", "<visual>")
            .replace(" <visual>", "<visual>")
        )
        # print(len(image_appeared_order))
        text = f"{text}<|endofchunk|>{self.tokenizer.eos_token}"
        # self.custom_logger.save_text_to_file(text)
        self.tokenizer.padding_side = "right"
        # suppose the max_lengeth of docunment is 4096
        text_tensor = self.tokenizer(
            text, max_length=4096, truncation=True, padding="max_length", return_tensors="pt"
        )
        non_pad_indices = (text_tensor["input_ids"] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[1]
        real_text_token_len = len(non_pad_indices)
        # print(real_text_token_len)
        selected_image_positions = (text_tensor["input_ids"] == self.media_token_id).nonzero(as_tuple=True)[1] # 227, 567
        # random select N(64/128/256) tokens with the image as anchor, we should sample as many images as possible
        extra_images = len(selected_image_positions) - max_num_images

        start_index = random.choice(selected_image_positions[:extra_images]) if extra_images > 0 else selected_image_positions[0]
        # start_index = 0 # open-flamingo implementation

        # start_position = start_index
        # end_position = start_index + num_selected_tokens        
        random_shift = random.randint(0, shift_region) # !!!- 3 is to prevent shift 128  
        start_position = start_index - random_shift
        end_position = start_position + num_selected_tokens

        if start_position < 0 or real_text_token_len < num_selected_tokens:
            start_position = 0
            end_position = num_selected_tokens
        elif end_position > real_text_token_len:
            start_position = real_text_token_len - num_selected_tokens
            end_position = real_text_token_len
        # Using range
        selected_indices = list(range(start_position, end_position))
        text_tensor["input_ids"] = text_tensor["input_ids"][0, selected_indices].unsqueeze(0)
        # text_tensor["input_ids"][:, 0] = self.bos_token_id # mark the first token as beginoftext
        text_tensor["attention_mask"] = text_tensor["attention_mask"][0, selected_indices].unsqueeze(0)
        if image_appeared_order is not None:
            selected_image_ixs = [image_appeared_order[i] for i, pos in enumerate(selected_image_positions) if pos in selected_indices]
        else:
            selected_image_ixs = [i for i, pos in enumerate(selected_image_positions) if pos in selected_indices]
        if len(selected_image_ixs) > max_num_images:
                selected_image_ixs = selected_image_ixs[:max_num_images]
        if len(selected_image_ixs) == 0:
            raise ValueError("No images in sample!")
        return text_tensor, selected_image_ixs

    
    def generate_clean_text_sequence_mmc4(self, sentences, matched_sentence_ixs, matched_sentence_scores, info, num_selected_tokens=64, image_sim_thresh=0.24, text_flip=False, max_num_images=5,  clean_data_use_strategy="noisy_only", text_coherence=False):
        """
        w text_coherence:
            sample adjecent texts
            matched_sentence_ixs: the index of sentences that have matched images, if may be disordered, e.g. [4, 2], 
            means the first image is matched with the 4th sentence, the second image is matched with the 2nd sentence
            still have bug, sometimes matched_sentece_ixs is not null, but the returned selected_image_ixs is null
        wo text_coherence:
            this is simple, random select k images and generate text sequence.
            But for low simlarity images, we may need to use the generated caption to replace the original caption if  clean_data_use_strategy == "low_simlarity".
            return:
                selected_image_ixs: the index of images that are selected, e.g. [0, 2] means the first and third image in the original image list
            remain question:
                1. the matched_sentece_ixs may be repetated, like [0, 0, 2, 2, 1, 3], now we just ignore this
                2. do not consider text_flip when wo text_coherence
                3. open_flamingo: # avoid the situation where there's one <visual> token and it's at the end # 50% chance of keeping single image samples
                4. open_flamingo: include linear_sum_assignment
        return:
            selected_image_ixs: the index of images that are selected, e.g. [0, 2] means the first and third image in the original image list
        """
        assert len(matched_sentence_ixs) > 0
        text = "<s>"
        text += " " if random.random() <= 0.5 else ""
        text_flip_flag = text_flip and random.random() <= 0.5
        # adding image to the right of text
        image_appeared_order = [] # record the order of image appeared in the text ()
        # follow A.3.2 in flamingo, random input text before or after image (with highest matching score)
        non_visual_count = 0
        if text_coherence:
            if text_flip: # A.3.2 in flamingo
                if text_flip_flag:
                    for i, sentence in enumerate(sentences):
                        text += f"{sentence} "
                        for index, j in enumerate(matched_sentence_ixs):
                            if i == j and matched_sentence_scores[index] > image_sim_thresh:
                                text += "<|endofchunk|><visual>"
                                image_appeared_order.append(index)
                else:
                    for i, sentence in enumerate(sentences):
                        for index, j in enumerate(matched_sentence_ixs):
                            if i == j and matched_sentence_scores[index] > image_sim_thresh:
                                text += "<|endofchunk|><visual>"
                                image_appeared_order.append(index)
                        text += f"{sentence}"
            else:
                for i, sentence in enumerate(sentences):
                    non_visual_count += 1
                    padding_sentence = sentence
                    for index, j in enumerate(matched_sentence_ixs):
                        if clean_data_use_strategy == "low_simlarity":
                            if i == j:
                                image_appeared_order.append(index)
                                text += "<|endofchunk|><visual>"
                                if matched_sentence_scores[index] > image_sim_thresh:
                                    padding_sentence = sentence
                                else:
                                    if 'generated_caption' in info["image_info"][index]:
                                        generated_text_lists = info["image_info"][index]['generated_caption'].split("<EOC>")
                                        padding_sentence = gen_mixed_caption(sentence, generated_text_lists, clean_data_use_strategy="clean_only")
                                    else:
                                        padding_sentence = sentence
                                non_visual_count = 0
                        else:
                            if i == j and matched_sentence_scores[index] > image_sim_thresh:
                                image_appeared_order.append(index)
                                if 'generated_caption' in info["image_info"][index]:
                                    generated_text_lists = info["image_info"][index]['generated_caption'].split("<EOC>")
                                    padding_sentence = gen_mixed_caption(sentence, generated_text_lists, clean_data_use_strategy=clean_data_use_strategy)
                                else:
                                    padding_sentence = sentence
                                text += f"<|endofchunk|><visual>"
                                non_visual_count = 0
                    # prevent the cases that two images are too far
                    if non_visual_count >= 2:
                        if random.random() >= min(0.8, 0.25 * non_visual_count):
                            text += f"{padding_sentence}"
                    else:
                        text += f"{padding_sentence}"
        else:
            for i, sentence in enumerate(sentences):
                for index, j in enumerate(matched_sentence_ixs):
                    # do not drop samples possibly
                    if clean_data_use_strategy == "low_simlarity":
                        if i == j:
                            image_appeared_order.append(index)
                            text += "<|endofchunk|><visual>"
                            if matched_sentence_scores[index] > image_sim_thresh:
                                text += f"{sentence}"
                            else:
                                if 'generated_caption' in info["image_info"][index]:
                                    generated_text_lists = info["image_info"][index]['generated_caption'].split("<EOC>")
                                    mixed_text = gen_mixed_caption(sentence, generated_text_lists, clean_data_use_strategy="clean_only")
                                else:
                                    mixed_text = sentence
                                text += f"{mixed_text}"
                    else:
                        if i == j and matched_sentence_scores[index] > image_sim_thresh:
                            image_appeared_order.append(index)
                            if 'generated_caption' in info["image_info"][index]:
                                generated_text_lists = info["image_info"][index]['generated_caption'].split("<EOC>")
                                mixed_text = gen_mixed_caption(sentence, generated_text_lists, clean_data_use_strategy=clean_data_use_strategy)
                            else:
                                mixed_text = sentence
                            if text_flip_flag:
                                text += f"{mixed_text}<|endofchunk|><visual>"
                            else:
                                text += f"<|endofchunk|><visual>{mixed_text}"
        text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
        if len(image_appeared_order) == 0:
            raise ValueError("No images in mmc4 sample")
        text_tensor, selected_image_ixs = self.sample_sequence_from_document(text, num_selected_tokens=num_selected_tokens, max_num_images=max_num_images, image_appeared_order=image_appeared_order)
        return text_tensor, selected_image_ixs, text

    def generate_clean_text_sequence_obelics(self, sentences, info, scores=None, num_selected_tokens=64, ideal_imgs=-1, image_sim_thresh=0.20, clean_data_use_strategy="noisy_only"):
        """
        1. sub-sample each sentences to fixed length if ideal_imgs is not -1.
        2. concat all sentences with <|endofchunk|> and <visual> token
        3. random select num_selected_tokens tokens with the image as anchor
        4. since a lot of sample have only two sentence (one None) we try to attend to the right, flip such sentences
        5. The scores is cross matching. For example, if we have 3 images and 2 sentences, then the scores is 3x2. 
        Remain question:
            1. if need to add a <endofchunk> token at the end of the final token?
            2. obelics is irregular, for example Text1, Text2, Image1 ...or Text1, Image1, Image2, Image3. How to process this?
        """
        text = "<s>"
        text += " " if random.random() <= 0.5 else ""
        # Image count tracker
        img_count = 0
        image_indexs, matched_sentence_ixs, matched_sentence_scores = select_image_index_from_score(scores, len(sentences), len(sentences))
        
        for i in range(len(sentences)):
            sentence = sentences[i]
            if sentence:
                text += f"{sentence}"
            else:
                # copy the most similar text if larger than threshold. Otherwise, use the generated caption
                if i not in image_indexs:
                    continue
                text += "<|endofchunk|><visual>"
                img_count += 1      
                if clean_data_use_strategy == "low_simlarity":
                    if i in image_indexs and matched_sentence_scores[image_indexs.index(i)] > image_sim_thresh:
                        text += f"{sentences[matched_sentence_ixs[image_indexs.index(i)]]}"
                    else:
                        if 'generated_caption' in info:
                            # some images may be broken or not download success
                            if info['generated_caption'][i] == None:
                               text += ""
                            else:
                                generated_text_lists = info['generated_caption'][i].split("<EOC>")
                                mixed_text = gen_mixed_caption(sentence, generated_text_lists, clean_data_use_strategy="clean_only")
                                text += f"{mixed_text}"       
                else:
                    text += f"{sentences[matched_sentence_ixs[image_indexs.index(i)]]}"       

        if img_count == 0:
            raise ValueError("No images in obelics sample!")

        text_tensor, selected_image_ixs = self.sample_sequence_from_document(text, num_selected_tokens=num_selected_tokens, max_num_images=ideal_imgs)
        return text_tensor, image_indexs, selected_image_ixs
    
    def generate_clean_text_sequence_cc3m(self, sentences):
        """
        Quite simple, concat all texts together and add <s> token at the begining
        """
        text = "<s>"
        text += " " if random.random() <= 0.5 else ""
       
        for _, sentence in enumerate(sentences):
            text += f"<visual>{sentence}<|endofchunk|>"
        text = (
            text.replace(" <|endofchunk|>", "<|endofchunk|>")
            .replace("<visual> ", "<visual>")
            .replace(" <visual>", "<visual>")
        )
        text = f"{text}{self.tokenizer.eos_token}"
        self.tokenizer.padding_side = "right"
        # suppose the max_lengeth of docunment is 128
        text_tensor = self.tokenizer(
            text, max_length=self.dataset_params['inter_vid_txt']['MAX_NUM_TOKENS'], truncation=True, padding="max_length", return_tensors="pt"
            )
        return text_tensor
    

    def read_selected_images(self, info, image_indexs, max_num, flip=False, dataset="mmc4"):
        """
        read the image and padding zero if not enough images
        """
        images = []
        if dataset in ["obelics", "cc3m_interlevel"]:
            filtered_info = info
            # filtered_info = filtered_info[::-1] if flip else filtered_info
        for ix in image_indexs:
            if dataset in ["obelics", "cc3m_interlevel"]:
                image_base64 = filtered_info[int(ix)]
            elif dataset == "mmc4":
                image_base64 = info["image_info"][ix]["image_base64"]
            else:
                raise ValueError("dataset not defined")
            rawbytes = base64.b64decode(image_base64)
            try:
                image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
            except Exception as e:
                self.custom_logger.info(e)
                continue
            images.append(image)
        if len(images) == 0:
            raise ValueError("No images in {} sample when read images".format(dataset))
        images_tensors = self.preprocess_image(images)
        keep_ixs = range(min(len(images_tensors), max_num))
        images_tensors = images_tensors[keep_ixs]
        if len(images_tensors) < max_num:
            zero_padding = torch.zeros(
                (max_num - len(images_tensors), 3, 224, 224), dtype=torch.float
            )
            images_tensors = torch.cat((images_tensors, zero_padding), dim=0)
        return images_tensors
    
    def define_learnable_mask(self, text_tensor, mask_strategy="mmc4"):
        """
        for mmc4 dataset, we mask the text before the first image and also the text after the <eoc>
        """
        labels = text_tensor["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        for i in range(labels.shape[0]):
            label_idx = 0
            # follow flamingo, let the text in the left of the first <visual> token not learnable
            while (                                                                   
                label_idx < labels.shape[1] and labels[i][label_idx] != self.media_token_id 
            ):
                labels[i][label_idx] = -100
                label_idx += 1

            endofchunk_idxs = torch.where(labels[i] == self.endofchunk_token_id)[0]
            for endofchunk_idx in endofchunk_idxs:
                token_idx = endofchunk_idx + 1
                while (
                    token_idx < labels.shape[1]
                    and labels[i][token_idx] != self.media_token_id
                ):
                    labels[i][token_idx] = -100
                    token_idx += 1
        labels[labels == self.media_token_id] = -100
        labels[labels == self.bos_token_id] = -100
        # get index of all endofchunk tokens in the sequence, MASK THE SENTENCE BESIDES <visual>TEXT<EOS>
        # for example, the end of sentence part.
        return labels

    def mmc4_optimize_assignment(self, info, disturb=True):
        """
        This function is used to match the image and text in the mmc4 dataset.
        If set 0.025(standard dev), 95% of the similarity matrix will be in the range of [-0.05, 0.05]
        Disturb is used to disturb the similarity matrix, so that we can get different matching results.
        """
        sim_matrix = info["similarity_matrix"]
        valid_image_indices = []
        for i, sample_image in enumerate(info["image_info"]):
            valid_image_indices.append(i)

        sim_matrix = np.array(sim_matrix)  # of shape images x sentences
        if disturb:
            disturb_matrix = np.random.normal(0, 0.025, sim_matrix.shape)
            disturb_matrix = np.clip(disturb_matrix, -0.08, 0.08)
            sim_matrix = sim_matrix + disturb_matrix
        
        sim_matrix = sim_matrix[valid_image_indices]
        cost_matrix = -sim_matrix
        image_indices, sentence_indices = linear_sum_assignment(cost_matrix)

        matched_sentence_ixs = sentence_indices
        matched_sentence_scores = sim_matrix[image_indices, sentence_indices]
        del valid_image_indices
        del disturb_matrix
        del sim_matrix
        return matched_sentence_ixs, matched_sentence_scores


    def preprocess_interleaved(self, sample, text_coherence=False, clean_data_use_strategy="noisy_only", balanced_sampling=True, cc3m_strategy="random"):
        """
        this function implement:
        (1). process image and text into long text sequence, and randomly select N tokens; Return selected image list
        (2). read selected images from the image list
        (3). define learnable mask for the selected text tokens
        """
        info = json.loads(sample)
        if "dataset" in info and info["dataset"] == "obelics":
            sentences = info["texts"]
            image_info = info["image_info"]
            # count non None image
            len_image_info = len([image for image in image_info if image is not None])
            if "score" in info:
                matched_sentence_scores = ast.literal_eval(info["score"])
            else:
                matched_sentence_scores = [["", 1.0] for _ in range(len(sentences)*len(image_info))]
            sample_flip = True if random.random() <= 0.5 else False
            # prevent the situation that only one image and one text, text is at last, then flip all labels will be -100
            if len(sentences) == 2 and sentences[1] is None:
                sample_flip = False
            if len(sentences) == 2 and sentences[0] is None:
                sample_flip = True
            if sample_flip:
                sentences = sentences[::-1]
                image_info = image_info[::-1]
                info['generated_caption'] = info['generated_caption'][::-1] if 'generated_caption' in info else None
                matched_sentence_scores = flip_scores(matched_sentence_scores, len(sentences), len(image_info))
            sentences = select_obelics_subsampled_text(sentences, len_image_info, num_selected_tokens=self.dataset_params['inter_img_txt']['MAX_NUM_TOKENS_OBELICS'], ideal_num_images=self.dataset_params['inter_img_txt']['MAX_NUM_IMAGES_OBELICS'])
            text_tensor, image_indexs, selected_image_ixs = self.generate_clean_text_sequence_obelics(sentences, info, scores=matched_sentence_scores, num_selected_tokens=self.dataset_params['inter_img_txt']['MAX_NUM_TOKENS_OBELICS'], ideal_imgs=self.dataset_params['inter_img_txt']['MAX_NUM_IMAGES_OBELICS'], image_sim_thresh=self.dataset_params['inter_img_txt']['SIM_THRESHOLD_OBELICS'], clean_data_use_strategy=clean_data_use_strategy)
            # image_indexs is real index and selected is realitive index, so we need to map them
            selected_image_ixs = [image_indexs[i] for i in selected_image_ixs]
            images_tensors = self.read_selected_images(image_info, selected_image_ixs, self.dataset_params['inter_img_txt']['MAX_NUM_IMAGES_OBELICS'], flip=sample_flip, dataset="obelics")
            labels = self.define_learnable_mask(text_tensor, mask_strategy="obelics")
        elif "dataset" in info and info["dataset"] == "cc3m_interlevel":
            sentences = info["texts"]
            images = info["image_info"]
            max_imgs = self.dataset_params['inter_img_txt']['MAX_NUM_IMAGES_CC3M']
            if cc3m_strategy == "adjacent":
                start = random.randint(0, len(info) - max_imgs)
                sampled_sentences = sentences[start:start + max_imgs]
                selected_image_ixs = list(range(start, start + max_imgs))
            elif cc3m_strategy == "random":
                # get random index
                sampled_indices = random.sample(range(len(info)), max_imgs)
                sampled_sentences = [sentences[i] for i in sampled_indices]
                selected_image_ixs = sampled_indices
            else:
                Exception("strategy not implemented")

            sampled_sentences = select_cc3m_subsampled_text(sampled_sentences, len(selected_image_ixs), selected_image_ixs, num_selected_tokens=self.dataset_params['inter_img_txt']['MAX_NUM_TOKENS_CC3M'], ideal_num_images=self.dataset_params['inter_img_txt']['MAX_NUM_IMAGES_CC3M'])
            text_tensor = self.generate_clean_text_sequence_cc3m(sampled_sentences)
            images_tensors = self.read_selected_images(images, selected_image_ixs, self.dataset_params['inter_img_txt']['MAX_NUM_IMAGES_CC3M'], dataset="cc3m_interlevel")
            labels = self.define_learnable_mask(text_tensor)
        else:
            sentences = info["text_list"]
            # optimal assignment to match images and sentences
            matched_sentence_ixs, matched_sentence_scores = self.mmc4_optimize_assignment(info)
            # matched_sentence_ixs = [sample_image["matched_text_index"] for sample_image in info["image_info"]]
            # matched_sentence_scores = [sample_image["matched_sim"] for sample_image in info["image_info"]]
            # then sample the text according to the matched sentence index
            sentences = select_mmc4_subsampled_text(sentences, len(info["image_info"]), matched_sentence_ixs, num_selected_tokens=self.dataset_params['inter_img_txt']['MAX_NUM_TOKENS_MMC4'], ideal_num_images=self.dataset_params['inter_img_txt']['MAX_NUM_IMAGES_MMC4'])
            text_tensor, selected_image_ixs, text = self.generate_clean_text_sequence_mmc4(sentences, matched_sentence_ixs, matched_sentence_scores, info, num_selected_tokens=self.dataset_params['inter_img_txt']['MAX_NUM_TOKENS_MMC4'], image_sim_thresh=self.dataset_params['inter_img_txt']['SIM_THRESHOLD_MMC4'],  max_num_images=self.dataset_params['inter_img_txt']['MAX_NUM_IMAGES_MMC4'],  clean_data_use_strategy=clean_data_use_strategy, text_coherence=text_coherence)
            images_tensors = self.read_selected_images(info, selected_image_ixs, self.dataset_params['inter_img_txt']['MAX_NUM_IMAGES_MMC4'], dataset="mmc4")
            labels = self.define_learnable_mask(text_tensor)
        text_tensor["labels"] = labels
        return {"visual": images_tensors, "input_ids": text_tensor["input_ids"], 
                "attention_mask":  text_tensor["attention_mask"], "labels": text_tensor["labels"]}

    def define_learnable_mask_of(self, text_tensor):
        """
        Define learnable mask for the selected text tokens (from open-flamingo)
        """
        labels = text_tensor["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        for i in range(labels.shape[0]):
            # remove loss for any token before the first <visual> token
            label_idx = 0
            while (
                label_idx < labels.shape[1] and labels[i][label_idx] != self.media_token_id
            ):
                labels[i][label_idx] = -100
                label_idx += 1

            # get index of all endofchunk tokens in the sequence
            endofchunk_idxs = torch.where(labels[i] == self.endofchunk_token_id)[0]
            for endofchunk_idx in endofchunk_idxs:
                token_idx = endofchunk_idx + 1
                while (
                    token_idx < labels.shape[1]
                    and labels[i][token_idx] != self.media_token_id
                ):
                    labels[i][token_idx] = -100
                    token_idx += 1

        labels[labels == self.media_token_id] = -100
        return labels

    def generate_interleaved_video_text_sequence(self, sentences):
        text = "<s>"
        for _, sentence in enumerate(sentences):
            text += f"<visual>{sentence}<|endofchunk|>"
        text = (
            text.replace(" <|endofchunk|>", "<|endofchunk|>")
            .replace("<visual> ", "<visual>")
            .replace(" <visual>", "<visual>")
        )
        text = f"{text}{self.tokenizer.eos_token}"
        self.tokenizer.padding_side = "right"
        # suppose the max_lengeth of docunment is 128
        text_tensor = self.tokenizer(
            text, max_length=self.dataset_params['inter_vid_txt']['MAX_NUM_TOKENS'], truncation=True, padding="max_length", return_tensors="pt"
            )
        return text_tensor


    def generate_instruction_text_sequence(self, raw_text, data_type="image"):
        text =  f"{raw_text}<|endofchunk|>"
        text = (
            text.replace("<image>", "<|visual|>")
            .replace("<visual> ", "<visual>")
            .replace(" <visual>", "<visual>")
        )
        text = f"{text}{self.tokenizer.eos_token}"
        self.tokenizer.padding_side = "right"
        # suppose the max_lengeth of docunment is 128
        if data_type == "video":
            text_tensor = self.tokenizer(
                text, max_length=self.dataset_params['img_instruct']['MAX_NUM_TOKENS'], truncation=True, padding="max_length", return_tensors="pt"
                )
        elif data_type == "image":
            text_tensor = self.tokenizer(
                text, max_length=self.dataset_params['vid_instruct']['MAX_NUM_TOKENS'], truncation=True, padding="max_length", return_tensors="pt"
                )
        else:
            raise ValueError("data_type not defined")
        return text_tensor

    def define_interleave_video_learnable_mask(self, text_tensor):
        labels = text_tensor["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[labels == self.media_token_id] = -100
        return labels

    def define_instruction_learnable_mask(self, text_tensor):
        """
        The text is like: <s>HUMAN:<text1>,GPT:<text2>,<|endofchunk|><eos>
        Give -100 for human text, media token id and pad_token_id token
        """
        labels = text_tensor["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[labels == self.media_token_id] = -100

        # remove loss for any token for human text
        for i in range(labels.shape[0]):
            for label_idx in range(labels.shape[1]):
                if labels[i][label_idx] == self.human_token_id:
                    labels[i][label_idx] = -100
                    # find the next comma
                    while label_idx < labels.shape[1] and labels[i][label_idx] != self.gpt_token_id:
                        labels[i][label_idx] = -100
                        label_idx += 1
        labels[labels == self.gpt_token_id] = -100
        labels[:, 0] = -100 # mark the first token as beginoftext

        return labels

    def preprocess_interleaved_video_wds(self, data, video_speed=5, mode='train', max_clips=3, strategy='adjacent'):
        """
        sample: video_name	clips
        visual is 6D input: (batch_size, Image in same chunk, Time, num_channels, height, width)
        video_speed: to cut the video size, we can speed up the video, for example, 5 means 5x speed up
        """
        video_reader, info = data
        info = info["clips"]
        info = json.loads(info)
        # print(f"video reader is: {video_reader}, info is: {info}, info len is: {len(info)}" )
        # if the info is too short, we pad it with the last clip
        if len(info) < max_clips:
            info = info + [info[-1]] * (max_clips - len(info))
        if strategy == "adjacent":
            start = random.randint(0, len(info) - max_clips)
            sampled_clips = info[start:start + max_clips]
        elif strategy == "random":
            sampled_clips = random.sample(info, max_clips)
        else:
            Exception("strategy not implemented")
        if len(sampled_clips) < max_clips:
            raise Exception("too few clips!")
        clip_texts = []
        # optimize this part, it will load video multiple times
        time_array = []
        for clip in sampled_clips:
            start, end = clip['clip'].split(' - ')
            # if use speed up howto100m with 5, the time should be divided by 5
            start_sec = sum(int(x) * 60 ** i for i, x in enumerate(reversed(start.split(':'))))
            end_sec = sum(int(x) * 60 ** i for i, x in enumerate(reversed(end.split(':'))))
            if start_sec > end_sec:
                start_sec = end_sec - 2
            time_array.append((float(start_sec)/video_speed, float(end_sec)/video_speed))
            clip_text = clip['caption']
            clip_texts.append(clip_text)
        video_tensor = read_frames_from_timestamps_and_path(video_reader, self.dataset_params['inter_vid_txt']['VIDEO_FRAMES'], time_array)
        video_tensor = video_augment(video_frame=self.dataset_params['inter_vid_txt']['VIDEO_FRAMES']*len(time_array), video_image_size=self.dataset_params['inter_vid_txt']['VIDEO_IMAGE_SIZE'], mode=mode)(video_tensor).permute(1, 0, 2, 3).unsqueeze(0) # cthw -> 1, tchw
        video_tensors = video_tensor.view(len(time_array), -1, video_tensor.shape[2], video_tensor.shape[3], video_tensor.shape[4])
        text_tensor = self.generate_interleaved_video_text_sequence(clip_texts)
        text_tensor['labels'] = self.define_interleave_video_learnable_mask(text_tensor)
        
        return [{"visual": video_tensors, "input_ids": text_tensor["input_ids"].squeeze(0), 
                "attention_mask":  text_tensor["attention_mask"].squeeze(0), "labels": text_tensor["labels"].squeeze(0)}]


    def preprocess_interleaved_video(self, video_path, info, video_speed=1, mode='train', max_clips=3, strategy='adjacent',  read_video_by_azfuse='False'):
        """
        sample: video_name	clips
        visual is 6D input: (batch_size, Image in same chunk, Time, num_channels, height, width)
        remain questions: when the video clips less than max_clips, how to deal with it?
        """
        if strategy == "adjacent":
            start = random.randint(0, len(info) - max_clips)
            sampled_clips = info[start:start + max_clips]
        elif strategy == "random":
            sampled_clips = random.sample(info, max_clips)
        else:
            Exception("strategy not implemented")
        clip_texts = []
        for clip in sampled_clips:
            start, end = clip['clip'].split(' - ')
            # if use speed up howto100m with 5, the time should be divided by 5
            start_sec = sum(int(x) * 60 ** i for i, x in enumerate(reversed(start.split(':'))))
            end_sec = sum(int(x) * 60 ** i for i, x in enumerate(reversed(end.split(':'))))
            video_tensor = read_frames_from_timestamps_ffmpeg(video_path, self.dataset_params['inter_vid_txt']['VIDEO_FRAMES'], mode=mode, start=float(start_sec)/video_speed, end=float(end_sec)/video_speed,
             read_video_by_azfuse=read_video_by_azfuse)
            video_tensor = video_augment(video_frame=self.dataset_params['inter_vid_txt']['VIDEO_FRAMES'], video_image_size=self.dataset_params['inter_vid_txt']['VIDEO_IMAGE_SIZE'], mode=mode)(video_tensor).permute(1, 0, 2, 3).unsqueeze(0) # cthw -> tchw
            clip_text = clip['caption']
            if 'video_tensors' in locals():
                video_tensors = torch.cat((video_tensors, video_tensor), dim=0)
            else:
                video_tensors = video_tensor
            clip_texts.append(clip_text)
        text_tensor = self.generate_interleaved_video_text_sequence(clip_texts)
        text_tensor['labels'] = self.define_interleave_video_learnable_mask(text_tensor)
        return {"visual": video_tensors, "input_ids": text_tensor["input_ids"].squeeze(0), 
                "attention_mask":  text_tensor["attention_mask"].squeeze(0), "labels": text_tensor["labels"].squeeze(0)}


    def preprocess_instruction(self, visual_tensors, text):
        """
        1. preprocess text, replace <image> with <visual>
        2. generate text label
        3. return trainable tensors
        """
        text_tensor = self.generate_instruction_text_sequence(text)
        text_tensor['labels'] = self.define_instruction_learnable_mask(text_tensor)
        return {"visual": visual_tensors.unsqueeze(0).unsqueeze(0), "input_ids": text_tensor["input_ids"].squeeze(0), 
                "attention_mask":  text_tensor["attention_mask"].squeeze(0), "labels": text_tensor["labels"].squeeze(0)}
