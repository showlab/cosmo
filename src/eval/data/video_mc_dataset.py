from torch.utils.data import Dataset
import json
import os
import pandas as pd
import random
from src.data.base.video_process import read_frames_gif, read_frames_decord_from_path
from src.data.base import video_augment


# tgif action and transition: {
#     "gif_name": "tumblr_nk172bbdPI1u1lr18o1_250",
#     "question": "What does the butterfly do 10 or more than 10 times ?",
#     "options": ["stuff marshmallow", "holds a phone towards face",
#                 "fall over", "talk", "flap wings"],
#     "answer": 4
# }

class VIDEOMCDataset(Dataset):
    def __init__(
        self,
        video_dir_path,
        annotations_path,
        split="train",
        num_frames=4,
        dataset_name="tgif_mc",
    ):
        print("Loading Video MC dataset...")
        self.video_dir_path = video_dir_path
        self.split = split
        self.dataset_name = dataset_name
        self.num_frames = num_frames
        self.metadata = None
        # test shot tasks without training, so set model to test for aug
        self.video_transform = video_augment(video_frame=self.num_frames, video_image_size=224, mode='test')
        self._load_metadata(annotations_path)

    def __len__(self):
        if self.dataset_name in ['msvd_qa', 'tvqa_qa']:
            return sum(1 for line in self.metadata)
        elif self.dataset_name in ['msrvtt_qa']:
            return len(self.metadata)
        return len(self.metadata)

    def get_raw_video(self, sample):
        abs_fp = self._get_video_path(sample)
        if 'tgif' in self.dataset_name:
            imgs, _, _ = read_frames_gif(abs_fp, self.num_frames, mode=self.split)
        else:
            imgs, _, _ = read_frames_decord_from_path(abs_fp, self.num_frames, mode=self.split)
        if imgs is None:
            raise Exception("Invalid img!", abs_fp)
        else:
            return imgs

        
    def _load_tgifmc_action_metadata(self, annotations_path):
        # download specific
        metadata_dir = annotations_path
        split_files = {
            'train': 'action_train_w_id.jsonl',
            'val': 'action_val_w_id.jsonl',  # action_val.jsonl
            'test': 'action_test_w_id.jsonl'  # no GT label for test set
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_json(os.path.join(metadata_dir, target_split_fp), lines=True)
        self.metadata = metadata

    def _load_tgifmc_transition_metadata(self, annotations_path):
        # download specific
        metadata_dir = annotations_path
        split_files = {
            'train': 'transition_train_w_id.jsonl',
            'val': 'transition_val_w_id.jsonl',  # transition_val.jsonl
            'test': 'transition_test_w_id.jsonl'  # no GT label for test set
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_json(os.path.join(metadata_dir, target_split_fp), lines=True)
        self.metadata = metadata

    def _load_msrvttmc_metadata(self, annotations_path):
        metadata_dir = annotations_path
        split_files = {
            'train': 'msrvtt_mc_test.jsonl',         # no train and test available, only for zero-shot
            'val': 'msrvtt_mc_test.jsonl',
            'test': 'msrvtt_mc_test.jsonl'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_json(os.path.join(metadata_dir, target_split_fp), lines=True)
        self.metadata = metadata

    
    def _load_lsmdcmc_metadata(self, annotations_path):
        metadata_dir = annotations_path
        split_files = {
            'train': 'LSMDC16_multiple_choice_train.csv',
            'val': 'LSMDC16_multiple_choice_test_randomized.csv',  # 'LSMDC16_multiple_choice_valid.csv',
            'test': 'LSMDC16_multiple_choice_test_randomized.csv'
        }
        target_split_fp = split_files[self.split]
        print(os.path.join(metadata_dir, target_split_fp))
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t', header=None)
        self.metadata = metadata
        datalist = []
        for raw_id in range(len(metadata)):
            raw_d = metadata.iloc[raw_id]
            video_fp = raw_d[0]
            sub_path = video_fp.split('.')[0]
            remove = sub_path.split('_')[-1]
            sub_path = sub_path.replace('_'+remove,'/')
            rel_video_fp = sub_path + video_fp + '.avi'
            options = [raw_d[idx] for idx in range(5, 10)]
            d = dict(
                id=video_fp,
                video_id=rel_video_fp,
                answer=raw_d[10] - 1 if self.split in ['val', 'test'] else 0,
                options=options,
                question_id=raw_id,
            )
            datalist.append(d)
        self.metadata = datalist
        print("load split {}, {} samples".format(self.split, len(self.metadata)))


    def _load_metadata(self, annotations_path):
        if self.dataset_name == "msvd_mc":
            self._load_msvdqa_metadata(annotations_path)
        elif self.dataset_name == "msrvtt_mc":
            self._load_msrvttqa_metadata(annotations_path)
        elif self.dataset_name == "tgif_mc_action":
            self._load_tgifmc_action_metadata(annotations_path)
        elif self.dataset_name == "tgif_mc_transition":
            self._load_tgifmc_transition_metadata(annotations_path)
        elif self.dataset_name == "lsmdc_mc":
            self._load_lsmdcmc_metadata(annotations_path)
        else:
            raise Exception(f"Unknown Video VQA dataset {self.dataset_name}")


    def _get_video_path(self, sample):
        if self.dataset_name in ['msrvtt_mc', 'tvqa_mc']:
            return os.path.join(self.video_dir_path, sample["video_id"] + '.mp4')
        elif self.dataset_name in ['tgif_mc_action', 'tgif_mc_transition']:
            return os.path.join(self.video_dir_path, sample["gif_name"] + '.gif')
        elif self.dataset_name in ['msvd_mc']:
            return os.path.join(self.video_dir_path, self.youtube_mapping_dict['vid' + str(sample["video_id"])] + '.avi')
        elif self.dataset_name in ['lsmdc_mc']:
            return os.path.join(self.video_dir_path, sample["video_id"])
        else:
            raise Exception(f"Unknown Video VQA dataset {self.dataset_name}")

    def get_question_anwers(self, sample):
        """
        lsmdc have no question available
        """
        if "lsmdc" in self.dataset_name:
            question = "Select the correct answer from the candidates."
        else:
            question = sample["question"]
        if self.dataset_name in ['tgif_mc_action', 'tgif_mc_transition']:
            answers = sample['options'][sample['answer']]
            # answers = sample['answer']
        else:
            answers = sample["answer"]
        return question, answers

    def get_candidate(self, sample):
        candidate = sample["options"]
        return candidate

    def get_video(self, sample):
        video = self.get_raw_video(sample)
        # do augmentation here
        return video

    def __getitem__(self, idx):
        if self.dataset_name in ['msrvtt_qa', 'tvqa_qa']:
            sample = self.metadata.iloc[idx]
        elif self.dataset_name in ['msvd_qa']:
            sample = self.metadata[idx].iloc[0]
        elif self.dataset_name in ['lsmdc_mc']:
            sample = self.metadata[idx]
        else:
            sample = self.metadata.iloc[idx]
        question, answer = self.get_question_anwers(sample)
        candidate = self.get_candidate(sample)
        video = self.get_video(sample)
        return {
            "video": video,
            "question": question,
            "answer": answer,
            "candidates": candidate,
            "question_id": sample['question_id']
        }

        # max_try = 3
        # for i in range(max_try):
        #     if self.dataset_name in ['msrvtt_qa', 'tvqa_qa']:
        #         sample = self.metadata.iloc[idx]
        #     elif self.dataset_name in ['msvd_qa']:
        #         sample = self.metadata[idx].iloc[0]
        #     elif self.dataset_name in ['lsmdc_mc']:
        #         sample = self.metadata[idx]
        #     else:
        #         sample = self.metadata.iloc[idx]
        #     try:
        #         question, answer = self.get_question_anwers(sample)
        #         candidate = self.get_candidate(sample)
        #         video = self.get_video(sample)
        #         return {
        #             "video": video,
        #             "question": question,
        #             "answer": answer,
        #             "candidates": candidate,
        #             "question_id": sample['question_id']
        #         }
        #     except Exception as e:
        #         if "tgif" in self.dataset_name:
        #             print(f"Error: {e} for sample {sample['gif_name']}")
        #         else:
        #             print(f"Error: {e} for sample {sample['video_id']}")
        #         idx = random.randint(0, len(self.metadata) - 1)