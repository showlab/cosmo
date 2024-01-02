import os
from torch.utils.data import Dataset
import json
import csv
from src.data.base.video_process import read_frames_decord_from_path, read_frames_from_timestamps_ffmpeg

class VideoCaptionDataset(Dataset):
    def __init__(
        self,
        video_dir_path,
        annotations_path,
        split,
        dataset_name="youcook2",
        num_frames=8,
    ):
        self.video_dir_path = video_dir_path
        self.annotations = []
        self.split = split
        self.dataset_name = dataset_name
        self.num_frames = num_frames
        # load from json file
        print("Loading captions from", annotations_path)
        with open(annotations_path, "r") as f:
            full_annotations = json.load(f)
        self.annotations = full_annotations['images']

    def __len__(self):
        return len(self.annotations)

    def _get_video_path(self, sample):
        if self.dataset_name in ['youcook2', 'tvc', 'msrvtt', 'msvd']:
            return os.path.join(self.video_dir_path, sample["file_name"])
        elif self.dataset_name in ['vatex']:
            return self.video_dir_path + sample["file_name"]
        else:
            raise Exception(f"Unknown Video Captioning dataset {self.dataset_name}")
        
    def get_raw_video(self, sample):
        abs_fp = self._get_video_path(sample)
        if self.dataset_name in ['youcook2', 'vatex', 'msrvtt', 'msvd']:
            imgs, idxs, vlen = read_frames_decord_from_path(abs_fp, self.num_frames, mode=self.split)
        elif self.dataset_name in ['tvc']:
            imgs = read_frames_from_timestamps_ffmpeg(abs_fp, self.num_frames, mode=self.split, start=float(sample['start']), end=float(sample['end']))
            # Transform imgs from C T H W to T, H, W, C
            imgs = imgs.permute(1, 2, 3, 0)
        if imgs is None:
            raise Exception("Invalid video!", abs_fp)
        else:
            return imgs
        
    def get_video(self, sample):
        video = self.get_raw_video(sample)
        # do augmentation here
        return video
    
    def __getitem__(self, idx):
        sample = self.annotations[idx]
        video = self.get_video(sample)
        caption = sample["caption"]
        return {
            "video": video,
            "caption": caption,
            "video_id": sample["id"]
        }
