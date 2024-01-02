import os
import pandas as pd
import json
import random
import io
try:
    from azfuse import File
except Exception as e:
    print("azfuse not supported on this cluster, use local file system instead")
from .base_dataset import BaseDataset
from .utils import split_data_by_node, shuffle_list
from .base.video_process import read_frames_from_timestamps_ffmpeg

class InterleaveVideoTextDataset(BaseDataset):
    def __init__(self, split, data_path, batch_size, tokenizer, image_processor, video_processor, dataset_params, custom_logger):
        super().__init__(split, data_path, batch_size, tokenizer, image_processor, video_processor, dataset_params, custom_logger)
        self.metadata = None
        self.read_video_by_azfuse = False
        self._load_metadata(data_path.strip())
        self.custom_logger.info(f"{self.split} Interleave Video Text Dataset num_samples: {len(self.metadata)}")
        if self.split_data_by_node:
            self.node_metadata = split_data_by_node(self.metadata)
        else:
            self.node_metadata = self.metadata
        if self.dataset_params['vid_txt']['MAX_SAMPLES'] == -1:
            print("Using all samples for interlevel video text")
        else:
            self.node_metadata = self.node_metadata[:self.dataset_params['inter_vid_txt']['MAX_SAMPLES']]
        # [{'clip': '0:00:00 - 0:00:05', 'caption': 'The video opens with a white background featuring the word "elflow" in green letters, occupying a central position.'}, ...]
        self.node_metadata['clips'] = self.node_metadata['clips'].apply(json.loads)
        if self.use_azfuse and self.inter_vwt_tar_pre_cache:
            self.read_video_by_azfuse = True
            all_video_path = self._get_all_video_path()
            File.prepare(all_video_path[:int(self.vwt_pre_cache_ratio * len(all_video_path))])
            print("Prepared {} videos for video text dataset!!".format(int(self.inter_vwt_pre_cache_ratio * len(all_video_path))))

    def _load_metadata(self, data_path):
        if self.use_azfuse and self.inter_vwt_tar_pre_cache:
            with File.open(data_path, 'r') as fp:
                content = fp.read()
                # Now that you have the content as a string, use StringIO to treat that string as a file-like object
                buffer = io.StringIO(content)
                self.metadata = pd.read_csv(buffer, sep='\t', header=0)
        else:
            data_path = data_path.replace('/azfuse', '')
            self.metadata = pd.read_csv(data_path, sep='\t', header=0)

    def _get_all_video_path(self):
        video_paths = [self._get_video_path(self.node_metadata.iloc[i]) for i in range(len(self.node_metadata))]
        return video_paths
    
    def _get_video_path(self, sample):
        """
        already transform all videos into mp4 format
        """
        # video_data_root = "/home/jinpeng/blob/vigstandard_data/v-jinpewang" 
        video_data_root = "/storage/v-jinpewang" 
        # /storage
        abs_fp = os.path.join(f"{video_data_root}/dataset/howto100m/videos_256", sample['path'])
        if self.read_video_by_azfuse:
            abs_fp = abs_fp.replace(f'{video_data_root}', f'{video_data_root}/azfuse')
        video_format = abs_fp.split('.')[-1]
        if video_format == 'webm':
            abs_fp = abs_fp.replace('webm', 'mp4')
        return abs_fp

    def __len__(self):
        return len(self.node_metadata)

    def __getitem__(self, index):
        return self.get_suite(index)
    

    def get_suite(self, index):
        sample = self.node_metadata.iloc[index]
        result = False
        video_path =  self._get_video_path(sample)
        video_speed = 5 # 1 is normal
        info = sample['clips']

        # ret = self.preprocess_interleaved_video_fn(video_path, info, video_speed, max_clips=self.dataset_params['inter_vid_txt']['VIDEO_SAMPLED_CLIPS'], mode=self.split,  read_video_by_azfuse=self.read_video_by_azfuse)
        while result is False:
            try:
                video_path =  self._get_video_path(sample)
                video_speed = 5 # 1 is normal
                info = sample['clips']
                ret = self.preprocess_interleaved_video_fn(video_path, info, video_speed, max_clips=self.dataset_params['inter_vid_txt']['VIDEO_SAMPLED_CLIPS'], mode=self.split,  read_video_by_azfuse=self.read_video_by_azfuse)
                result = True
            except Exception as e:
                self.custom_logger.info(f"Error while read file idx {sample[0]} -> {e}")
                index = random.randint(0, len(self.node_metadata) - 1)
                sample = self.node_metadata.iloc[index]
        return ret