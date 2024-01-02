"""
unfortunately :(, the webdataset do not support video decoding very well and memory cost is high,
so we use decord to decode video and use pytorch dataloader to load data
Original from: https://github.com/showlab/all-in-one/blob/main/AllInOne/datasets/video_base_dataset.py
"""
import pandas as pd
import random
import io
from subprocess import Popen, PIPE
try:
    from azfuse import File
except Exception as e:
    print("azfuse not supported on this cluster, use local file system instead")
from .base.video_process import read_frames_decord_from_path
from .base import raw_video_tuple_to_dict
from .base_dataset import BaseDataset
from .utils import split_data_by_node
from utils import *

class VideoTextDataset(BaseDataset):
    def __init__(self, split, data_path, batch_size, tokenizer, image_processor, video_processor, dataset_params, custom_logger):
        super().__init__(split, data_path, batch_size, tokenizer, image_processor, video_processor, dataset_params, custom_logger)
        self.node_metadata = None
        self._load_metadata(data_path.strip())
        self.custom_logger.info(f"{self.split} Video Text Dataset num_samples: {len(self.metadata)}")
        self.custom_logger.info("batch size for video text dataset is: {}".format(self.batch_size))
        
    def _load_metadata(self, data_path):
        self.read_video_by_azfuse = False
        self.custom_logger.info(f"Loading metadata from {data_path}")
        if self.use_azfuse:
            with File.open(data_path, 'r') as fp:
                content = fp.read()
                # Now that you have the content as a string, use StringIO to treat that string as a file-like object
                buffer = io.StringIO(content)
                self.metadata = pd.read_csv(buffer, sep='\t', header=0)
        else:
            self.metadata = pd.read_csv(data_path, sep='\t', header=0)
        if self.split_data_by_node:
            self.node_metadata = split_data_by_node(self.metadata)
        else:
            self.node_metadata = self.metadata
        if self.dataset_params['vid_txt']['MAX_SAMPLES'] == -1:
            print("Using all samples for video text")
        else:
            self.node_metadata = self.node_metadata[:self.dataset_params['vid_txt']['MAX_SAMPLES']]
        if self.use_azfuse and self.vwt_tar_pre_cache:
            self.read_video_by_azfuse = True
            all_video_path = self._get_all_video_path()
            File.prepare(all_video_path[:int(self.vwt_pre_cache_ratio * len(all_video_path))])
            print("Prepared {} videos for video text dataset".format(int(self.vwt_pre_cache_ratio * len(all_video_path))))

    def __len__(self):
        return len(self.node_metadata)

    def __getitem__(self, index):
        return self.get_suite(index)
    
    def _get_all_video_path(self):
        video_paths = [self._get_video_path(self.node_metadata.iloc[i])[0] for i in range(len(self.node_metadata))]
        return video_paths

    def _get_video_path(self, sample):
        abs_fp = sample[0]
        rel_fp = sample[0].split('/')[-1]
        # read from other blob only
        if self.read_video_by_azfuse:
            abs_fp = abs_fp.replace('/storage', '/storage/azfuse')
        return abs_fp, rel_fp

    def get_raw_video(self, sample):
        abs_fp, rel_fp = self._get_video_path(sample)
        video, idxs, vlen = read_frames_decord_from_path(abs_fp, num_frames=self.dataset_params['vid_txt']['VIDEO_FRAMES'], read_video_by_azfuse=self.read_video_by_azfuse)
        if video is None:
            raise Exception("Invalid video!", rel_fp)
        else:
            return video
        
    def get_text(self, sample):
        """
        videopath, caption or
        videopath, caption,	gen_caption
        """
        if len(sample) > 2:
            text = sample[1] + '\t' + sample[2]
        else:
            text = sample[1]
        return self.preprocess_video_text_fn([text], padding_rule="max_length")


    def get_video(self, sample):
        video = self.get_raw_video(sample)
        processed_video_tensor = self.preprocess_video_fn(video.unsqueeze(0).unsqueeze(0))
        return processed_video_tensor.squeeze(0)

    def get_suite(self, index):
        result = None
        while result is None:
            sample = self.node_metadata.iloc[index]
            # print(sample)
            try:
                video_tensor = self.get_video(sample)
                txt_tensor = self.get_text(sample)
                ret = raw_video_tuple_to_dict([video_tensor, txt_tensor])
                result = True
            except Exception as e:
                self.custom_logger.info(f"Error while read file idx {sample[0]} -> {e}")
                index = random.randint(0, len(self.node_metadata) - 1)
        return ret