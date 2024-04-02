from typing import List
from PIL import Image
import torch
import torch.nn as nn
import json
import os
import numpy as np
import torch.nn.functional as F
from einops import repeat
from transformers.modeling_outputs import CausalLMOutputWithPast
from src.eval.models.eval_base_model import BaseEvalModel
from src.data.base_dataset import image_augment, video_augment

import importlib
import yaml
from safetensors.torch import load_file, load_model
# load different version of
with open('src/config/model_version/model_version.yaml') as f:
    config = yaml.safe_load(f)
model_version = config['load_model']['version']
model_module = importlib.import_module(f'multimodal_model.load_model{model_version}')
create_cosmo = getattr(model_module, 'create_cosmo')


def unwrap_model(model):
    """
    Unwrap a model from a DataParallel or DistributedDataParallel wrapper.
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    else:
        return model

def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    )
    return torch.index_select(x, dim, order_index.to(x.device))
        
class EvalModel(BaseEvalModel):
    """OpenFlamingo model evaluation.

    Attributes:
      model (nn.Module): Underlying Torch model.
      tokenizer (transformers.PreTrainedTokenizer): Tokenizer for model.
      device: Index of GPU to use, or the string "CPU"
    """

    def __init__(self, config, custom_logger, device):
        # load model
        # ========================================== model define ===========================================
        with open(config['deepspeed_config'], 'r') as f:
            deepspeed_config = json.load(f)
        self.cast_dtype = torch.float16
        # comment following lines if you do not use ompi on cluster
        os.environ["OMPI_COMM_WORLD_SIZE"] = "1"
        os.environ["OMPI_COMM_WORLD_RANK"] = "0"
        os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"] = "1"
        os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] = "0"

        self.model, self.image_processor, self.video_processor, self.tokenizer = create_cosmo(config["model_params"], None)
        if self.image_processor is None:
            self.image_processor = image_augment(mode="val")
        if type(self.image_processor) == list:
            self.image_processor = self.image_processor[1] # 0 train, 1 eval
        if self.video_processor is None:
            self.video_processor = video_augment(video_frame=4, video_image_size=224, mode='val')
        self.model = self.model.eval().to(device, dtype=self.cast_dtype, non_blocking=True)
        
        ckpt_path = config['general']['ckpt_path']
        print(f"!!load ckpt from: {ckpt_path}")
        if ckpt_path.endswith('.safetensors'): # HF checkpoint
            weights = load_file(ckpt_path)
            print(f"!!load ckpt from: {ckpt_path}")
            self.model.load_state_dict(weights, strict=False) # True
            # load_model(self.model, ckpt_path)
        else:
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            print(f"!!load ckpt from: {ckpt_path}")
            self.model.load_state_dict(ckpt, strict=False) # True
        self.device=device

        # # Set distinct pad_token_id and eos_token_id
        # self.tokenizer.pad_token = "<PAD>"
        # self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        # self.tokenizer.eos_token = "<EOS>"
        # self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)

        # # Initialize the DeepSpeed-Inference engine
        # ds_engine = deepspeed.init_inference(self.model,
        #                                 dtype= torch.float16,
        #                                 base_dir="/home/jinpeng/blob/vigstandard_data/v-jinpewang/experiments/VideoGPT4/mistral-7b_clip_16gpus_base_18m_7b_base/2023-11-20_19-52-39",
        #                                 replace_with_kernel_inject=True)
        # self.model = ds_engine.module

    def _prepare_text(
        self,
        batch: List[List[str]],
        padding="longest",
        truncation=True,
        max_length=2000,
    ):
        """
        Tokenize the text and stack them.
        Args:
            batch: A list of lists of strings.
        Returns:
            input_ids (tensor)
                shape (B, T_txt)
            attention_mask (tensor)
                shape (B, T_txt)
        """
        encodings = self.tokenizer(
            batch,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
            max_length=max_length,
        )
        input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
        input_ids = input_ids.to(self.device, dtype=torch.long, non_blocking=True)
        attention_mask = attention_mask.to(
            self.device, dtype=self.cast_dtype, non_blocking=True
        )
        return input_ids, attention_mask.bool()
    
    def _prepare_images(self, batch: List[List[torch.Tensor]]) -> torch.Tensor:
        """Preprocess images and stack them.

        Args:
            batch: A list of lists of images.

        Returns:
            A Tensor of shape
            (batch_size, images_per_example, frames, channels, height, width).
        """
        images_per_example = max(len(x) for x in batch)
        batch_images = None
        for iexample, example in enumerate(batch):
            for iimage, image in enumerate(example):
                preprocessed = self.image_processor(image).to(self.device)
                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed
        return batch_images

    def _prepare_images_tensors(self, batch_tensors: torch.Tensor) -> torch.Tensor:
        """Preprocess images and stack them.

        Args:
            batch_tensors: The batched image tensor (B, C, H, W).

        Returns:
            A Tensor of shape
            (batch_size, images_per_example, frames, channels, height, width).
        """
        batch_images = batch_tensors.unsqueeze(1).unsqueeze(1).to(self.device)
        return batch_images
    
    def _prepare_videos(self, batch: List[List[torch.Tensor]]) -> torch.Tensor:
        """Preprocess videos and stack them.

        Args:
            batch: A list of lists of videos. 

        Returns:
            A Tensor of shape
            (batch_size, videos_per_example, frames, channels, height, width).
        """
        videos_per_example = max(len(x) for x in batch)
        batch_videos = None
        for iexample, example in enumerate(batch):
            for ivideo, video in enumerate(example):
                # self.video_processor(s[0].permute(3, 0, 1, 2)).permute(1, 0, 2, 3).unsqueeze(0)
                preprocessed = self.video_processor(video.permute(3, 0, 1, 2)).permute(1, 0, 2, 3).to(self.device)
                if batch_videos is None:
                    batch_videos = torch.zeros(
                        (len(batch), videos_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_videos[iexample, ivideo, 0] = preprocessed
        # print(batch_videos.shape)
        if len(batch_videos.shape) != 7:
           print(f"batch_videos.shape: {batch_videos.shape}")
        batch_videos = batch_videos.squeeze(2)
        # print(batch_videos.shape)
        if len(batch_videos.shape) != 6:
           print(f"batch_videos.shape: {batch_videos.shape}")
        return batch_videos
    
    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        self.model.eval()

        self.tokenizer.padding_side = "left" # "right" ?
        encodings = self.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        with torch.inference_mode():
            outputs = self.model.generate(
                self._prepare_images(batch_images).to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
                input_ids.to(
                        self.device, dtype=torch.long, non_blocking=True
                    ),
                attention_mask=attention_mask.to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
                max_new_tokens=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )

        outputs = outputs[:, len(input_ids[0]) :]

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_video_outputs(
        self,
        batch_text,
        batch_videos,
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        self.model.eval()

        self.tokenizer.padding_side = "left" # "right" ?
        encodings = self.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        with torch.inference_mode():
            outputs = self.model.generate(
                self._prepare_videos(batch_videos).to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
                input_ids.to(
                        self.device, dtype=torch.long, non_blocking=True
                    ),
                attention_mask=attention_mask.to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
                max_new_tokens=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )

        outputs = outputs[:, len(input_ids[0]) :]

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    

    def encode_text(self, batch_text: List[str]) -> torch.Tensor:
        """
        Implement a clip style text encoder.
        Create a fake image input and use the model's forward pass.
        """
        self.model.eval()

        self.tokenizer.padding_side = "left"
        encodings = self.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Create a fake image input
        batch_images = [[Image.new("RGB", (224, 224)) for _ in batch_text]]

        with torch.inference_mode():
            text_embedding, _ = self.model.get_visual_text_embedding(
                self._prepare_images(batch_images).to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
                input_ids.to(
                        self.device, dtype=torch.long, non_blocking=True
                    ),
                attention_mask=attention_mask.to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
            )
        return text_embedding

    def encode_image(self, batch_images: torch.Tensor) -> torch.Tensor:
        """
        Implement a clip style image encoder.
        Create a fake text input and use the model's forward pass.
        """
        self.model.eval()

        # Create a fake text input
        batch_text = ["This is a fake text prompt." for _ in batch_images]
        self.tokenizer.padding_side = "left" # "right" ?
        encodings = self.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        labels = input_ids.clone()

        with torch.inference_mode():
            _, image_embedding = self.model.get_visual_text_embedding(
                self._prepare_images_tensors(batch_images).to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
                input_ids.to(
                        self.device, dtype=torch.long, non_blocking=True
                    ),
                attention_mask=attention_mask.to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
                labels=labels.to(
                        self.device, dtype=torch.long, non_blocking=True
                    ),
            )

        return image_embedding

    def get_embeddings(self, batch_text: List[str], batch_images: List[List[Image.Image]]) -> torch.Tensor:
        self.model.eval()

        self.tokenizer.padding_side = "left"
        encodings = self.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        labels = input_ids.clone()

        with torch.inference_mode():
            text_embedding, image_embedding = self.model.get_visual_text_embedding(
                self._prepare_images(batch_images).to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
                input_ids.to(
                        self.device, dtype=torch.long, non_blocking=True
                    ),
                attention_mask=attention_mask.to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
                labels=labels.to(
                        self.device, dtype=torch.long, non_blocking=True
                    ),
            )
        return text_embedding, image_embedding

    def get_video_mc_outputs(self, batch_text, batch_videos, batch_answer_list, max_generation_length: int, num_beams: int, length_penalty: float, num_ans_candidates=128):
        """
        This implemtion is mainly from BLIP2 (https://github.com/salesforce/LAVIS).
        1. Generate the first token of answers using decoder and select ${num_ans_candidates} most probable ones. 
        2. Then select answers from answer list, which start with the probable tokens.
        3. Lastly, use the selected answers as the ground-truth labels for decoding and calculating LM loss.
        Return the answers that minimize the losses as result.

        """
        self.model.eval()

        # padding_side？
        answer_candidates = self.tokenizer(
            batch_answer_list, padding="longest", return_tensors="pt"
        ).to(self.device)
        answer_candidates.input_ids[:, 0] = self.tokenizer.bos_token_id

        answer_ids = answer_candidates.input_ids
        answer_atts = answer_candidates.attention_mask


        self.tokenizer.padding_side = "left" # "right" ?
        encodings = self.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]


        with torch.inference_mode(): 
            start_output = self.model.generate(
                self._prepare_videos(batch_videos).to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
                input_ids.to(
                        self.device, dtype=torch.long, non_blocking=True
                    ),
                attention_mask=attention_mask.to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
                max_new_tokens=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(
            dim=1, index=answer_first_token
        )
        topk_probs, topk_ids = prob_first_token.topk(num_ans_candidates, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id, -100
        )

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, num_ans_candidates)
        question_atts = tile(question_atts, 0, num_ans_candidates)

        with torch.inference_mode(): 
            output = self.model.generate(
                self._prepare_videos(batch_videos).to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
                input_ids.to(
                        self.device, dtype=torch.long, non_blocking=True
                    ),
                attention_mask=attention_mask.to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
                max_new_tokens=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )

        num_ques = len(batch_text)
        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques, num_ans_candidates)

        max_topk_ids = log_probs_sum.argmax(dim=1)
        max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]

        answers = [batch_answer_list[max_id] for max_id in max_ids]

        return answers
    
    def get_rank_classifications(
            self,
            batch_text: List[str],
            batch_images: List[List[Image.Image]],
            all_class_names: List[str],
            use_cache: False,
            normalize_length: bool,
        ):
        """
        Returns a (B, |all_class_names|) tensor containing the logprobs for each class name.
        """
        batch_images = self._prepare_images(batch_images).to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    )
        ctx_input_ids, ctx_attention_mask = self._prepare_text(batch_text)

        # Cache the context
        if use_cache:
            # reserve the last token in the context for the main forward pass
            self.cache_media(
                input_ids=ctx_input_ids,
                vision_x=batch_images,
            )
            precomputed = self.__call__(
                vision_x=None,
                lang_x=ctx_input_ids,
                attention_mask=ctx_attention_mask,
                clear_conditioned_layers=False,
                use_cache=True,
            )
            precomputed_logits = precomputed.logits
            precomputed_pkvs = precomputed.past_key_values
        else:
            precomputed_pkvs = None

        # Loop through class names and get log-likelihoods
        # Note: if all classnames are one token, this code is redundant, since we could
        # get all logits after one pass. However, if there are multi-token classnames,
        # we need to loop through each classname separately.
        overall_probs = []
        for class_name in all_class_names:
            # Tokenize only the class name
            classname_tokens = self.tokenizer(
                class_name, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(self.device)
            assert classname_tokens.ndim == 2
            classname_tokens = repeat(
                classname_tokens, "b s -> (repeat b) s", repeat=len(batch_text)
            )
            num_tokens_in_classname = classname_tokens.shape[1]

            # Concatenate the class name tokens
            if not use_cache:
                _lang_x = torch.cat([ctx_input_ids, classname_tokens], dim=1)
                _attention_mask = torch.cat(
                    [
                        ctx_attention_mask,
                        torch.ones_like(classname_tokens).bool(),
                    ],
                    dim=1,
                )
                _vision_x = batch_images
            else:
                _lang_x = classname_tokens
                _attention_mask = None
                _vision_x = None

            # Call forward to get the logits
            outputs = self.__call__(
                vision_x=_vision_x,
                lang_x=_lang_x,
                attention_mask=_attention_mask,
                clear_conditioned_layers=(not use_cache),
                past_key_values=precomputed_pkvs,
            )

            # Get the logits of the classname
            # logits shape is either (B, num_tokens_in_classname, vocab_len) with use_cache
            # or (B, len(_lang_x), vocab_len) without use_cache
            # remember that the logits at index t on dim 1 correspond to predictions for the t+1st token
            logits = outputs.logits
            if use_cache:
                logits = torch.cat([precomputed_logits, logits], dim=1)

            logprobs = torch.log_softmax(logits, dim=-1)
            gen_probs = logprobs[
                :, -num_tokens_in_classname - 1 : -1, :
            ]  # (B, num_tokens_in_classname, vocab_len)
            gen_probs = torch.gather(
                gen_probs, 2, classname_tokens[:, :, None]
            ).squeeze(-1)

            # Aggregate over tokens in the classname
            if normalize_length:
                class_prob = torch.mean(gen_probs, dim=1)
            else:
                class_prob = torch.sum(gen_probs, dim=1)
            overall_probs.append(class_prob)  # (B, 1)

        self.uncache_media()
        overall_probs = torch.vstack(overall_probs).T.cpu()  # shape (B, num_classes)
        return overall_probs

    def __call__(
        self,
        lang_x: torch.Tensor,
        vision_x: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: torch.Tensor = None,
        clear_conditioned_layers: bool = False,
        use_cache: bool = False,
    ):
        """
        Calls the forward function of the model.
        Special logic to handle the case if past_key_values is not None:
            then lang_x is assumed to contain the tokens to be generated
            *excluding* the tokens already in past_key_values.
            We then repeatedly call forward, updating the past_key_values.
        """
        # standard forward pass
        if past_key_values is None:
            with torch.inference_mode():
                # with self.autocast():
                outputs = self.model(
                    vision_x=vision_x,
                    lang_x=lang_x,
                    attention_mask=attention_mask,
                    clear_conditioned_layers=clear_conditioned_layers,
                    past_key_values=past_key_values,
                    # use_cache=use_cache,
                )
            return outputs[0]

        # loop to handle updating past_key_values
        logits = []
        for token_idx in range(lang_x.shape[1]):
            _lang_x = lang_x[:, token_idx].reshape((-1, 1))
            if attention_mask is not None:
                _attention_mask = attention_mask[:, token_idx].reshape((-1, 1))
            else:
                _attention_mask = None

            with torch.inference_mode():
                # with self.autocast():
                outputs = self.model(
                    vision_x=vision_x,
                    lang_x=_lang_x,
                    attention_mask=_attention_mask,
                    clear_conditioned_layers=False,
                    past_key_values=past_key_values,
                    # use_cache=True,
                )

            past_key_values = outputs[0].past_key_values
            logits.append(outputs[0].logits)

        logits = torch.cat(logits, dim=1)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )
    
    def uncache_media(self):
        unwrap_model(self.model).uncache_media()

    def cache_media(self, input_ids, vision_x):
        unwrap_model(self.model).cache_media(input_ids=input_ids, vision_x=vision_x)

    def vqa_prompt(self, question, answer=None) -> str:
        # others: prompt = "<s> Instruction: provide an answer to the question. Use the image to answer."
        # vizwiz: Answer the questions based on the image when possible, otherwise say unanswerable.
        return f"<visual>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def video_mc_prompt(self, question, candidates, answer=None) -> str:
        # instruction = "Select a answer from candidates."
        # others: prompt = "<s> Instruction: provide an answer to the question. Use the image to answer."
        return f"<visual>Question:{question}, Candidates: {candidates}, Answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def caption_prompt(self, caption=None) -> str:
        # prompt = "Provide a short caption of the input image.
        return f"<visual>Output:{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"

    def classification_prompt(self, class_str=None) -> str:
        # It’s a conversation between a human, the user, and an intelligent visual AI, Bot. The user sends memes with text written on them, and Bot has to say whether the meme is hateful or not.
        # <Visual > if an image with written. "Is it hateful? Answer:"
        return f"<visual>A photo of a {class_str if class_str is not None else ''}{'<|endofchunk|>' if class_str is not None else ''}"

    def retrieval_prompt(self, caption=None) -> str:
        return f"<visual>A photo of a {caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"
    
    def generate_prompt(self, prompt=None) -> str:
        return f"<visual>{prompt if prompt is not None else ''}"
    
    def get_hateful_memes_prompt(self, text, label=None) -> str:
        return f"<image>is an image with: '{text}' written on it. Is it hateful? Answer:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"

    def idelics_vqa_prompt(self, question, answer=None) -> str:
        return f"<visual><human> Question:{question} \n Answer the question using a single word or phrase. <gpt> Answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"
    
    def idelics_vqa_prefix_prompt(self):
        prefix_prompt = ""
        return prefix_prompt

    def idelics_vizwiz_vqa_prompt(self, question, answer=None) -> str:
        return f"<visual><human> Question:{question} \nWhen the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase.  <gpt> Answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"
    
    def idelics_vizwiz_vqa_prefix_prompt(self):
        prefix_prompt = ""
        return prefix_prompt
    
    def idelics_hateful_memes_prompt(self, text, label=None) -> str:
        return f"<visual><human> is an image with: '{text}' written on it. Is it hateful? Answer:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"

    def idelics_hateful_memes_prefix_prompt(self):
        prefix_prompt = ""
        return prefix_prompt
    
    def idelics_caption_prompt(self, caption=None) -> str:
        return f"<visual><human>Provide a short description for this image. <gpt>{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"
    
    def idelics_caption_prefix_prompt(self):
        return ""