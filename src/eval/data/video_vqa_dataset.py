"""
part of the implementation comes from https://github.com/showlab/all-in-one/blob/main/AllInOne/datasets/msvdqa.py
Different from the original implementation rely on pre-defined "answer vocubulary",
we test a real open-ended setting where the model is not aware of the answer vocabulary.
"""

from torch.utils.data import Dataset
import json
import os
import pandas as pd
from src.data.base.video_process import read_frames_decord_from_path
from src.data.base import video_augment


class VIDEOVQADataset(Dataset):
    def __init__(
        self,
        video_dir_path,
        annotations_path,
        split="train",
        num_frames=8,
        dataset_name="msvd_qa",
    ):
        print("Loading Video VQA dataset...")
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
        imgs, idxs, vlen = read_frames_decord_from_path(abs_fp, self.num_frames, mode=self.split)
        if imgs is None:
            raise Exception("Invalid video!", abs_fp)
        else:
            return imgs
        
    def _load_msvdqa_metadata(self, annotations_path):
        metadata_dir = annotations_path
        split_files = {
            'train': 'msvd_train_qa_encode.json',
            'val': 'msvd_val_qa_encode.json',
            'test': 'msvd_test_qa_encode.json'
        }
        self.youtube_mapping_dict = dict()
        with open(os.path.join(metadata_dir, 'msvd_youtube_mapping.txt')) as f:
            lines = f.readlines()
            for line in lines:
                info = line.strip().split(' ')
                self.youtube_mapping_dict[info[1]] = info[0]
        target_split_fp = split_files[self.split]
        metadata = pd.read_json(os.path.join(metadata_dir, target_split_fp), lines=True)
        if self.metadata is None:
            self.metadata = metadata
        else:
            self.metadata.update(metadata)
        print("total {} samples for {} {}".format(self.__len__(), self.dataset_name, self.split))


    def _load_msrvttqa_metadata(self, annotations_path):
        metadata_dir = annotations_path
        split_files = {
            'train': 'msrvtt_qa_train_w_id.jsonl',
            'val': 'msrvtt_qa_val_w_id.jsonl',
            'test': 'msrvtt_qa_test_w_id.jsonl'
        }
        answer_fp = os.path.join(metadata_dir, 'msrvtt_train_ans2label.json')  # 1500 in total (all classes in train)
        # answer_fp = os.path.join(metadata_dir, 'msrvtt_qa_ans2label.json')  # 4539 in total (all classes in train+val+test)
        with open(answer_fp, 'r') as JSON:
            self.ans_lab_dict = json.load(JSON)
        
        target_split_fp = split_files[self.split]
        # path_or_buf=os.path.join(metadata_dir, target_split_fp)
        metadata = pd.read_json(os.path.join(metadata_dir, target_split_fp), lines=True)
        
        if self.metadata is None:
            self.metadata = metadata
        else:
            self.metadata.update(metadata)
        print("total {} samples for {} {}".format(self.__len__(), self.dataset_name, self.split))
    
    def _load_tgifqa_metadata(self, annotations_path):
        # download specific
        metadata_dir = annotations_path
        if self.data_split == "action":
            split_files = {
                'train': 'action_train.jsonl',
                'val': 'action_test.jsonl',  # action_val.jsonl
                'test': 'action_test.jsonl'  # no GT label for test set
            }
        elif self.data_split == "transition":
            split_files = {
                'train': 'transition_train.jsonl',
                'val': 'transition_test.jsonl',  # transition_val.jsonl
                'test': 'transition_test.jsonl'  # no GT label for test set
            }
        else:
            Exception("not support split")
        target_split_fp = split_files[self.split]
        metadata = pd.read_json(os.path.join(metadata_dir, target_split_fp), lines=True)
        self.metadata = metadata


    def _load_metadata(self, annotations_path):
        if self.dataset_name == "msvd_qa":
            self._load_msvdqa_metadata(annotations_path)
        elif self.dataset_name == "msrvtt_qa":
            self._load_msrvttqa_metadata(annotations_path)
        elif self.dataset_name == "tgif_qa":
            self._load_tgifqa_metadata(annotations_path)
        else:
            raise Exception(f"Unknown Video VQA dataset {self.dataset_name}")


    def _get_video_path(self, sample):
        if self.dataset_name in ['msrvtt_qa', 'tvqa_qa']:
            return os.path.join(self.video_dir_path, sample["video_id"] + '.mp4')
        elif self.dataset_name in ['tgif_qa']:
            return os.path.join(self.video_dir_path, sample["gif_name"] + '.gif')
        elif self.dataset_name in ['msvd_qa']:
            return os.path.join(self.video_dir_path, self.youtube_mapping_dict['vid' + str(sample["video_id"])] + '.avi')
        else:
            raise Exception(f"Unknown Video VQA dataset {self.dataset_name}")

    def get_question_anwers(self, sample):
        question = sample["question"]
        answers = sample["answer"]
        return question, answers

    def get_video(self, sample):
        video = self.get_raw_video(sample)
        # do augmentation here
        return video

    def __getitem__(self, idx):
        if self.dataset_name in ['msrvtt_qa', 'tvqa_qa']:
            sample = self.metadata.iloc[idx]
        elif self.dataset_name in ['msvd_qa']:
            sample = self.metadata[idx].iloc[0]
        else:
            sample = self.metadata.iloc[idx]
        question, answer = self.get_question_anwers(sample)
        video = self.get_video(sample)
        return {
            "video": video,
            "question": question,
            "answers": answer,
            "question_id": sample['id']
        }