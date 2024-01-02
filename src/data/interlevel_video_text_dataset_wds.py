"""
The wds will cache the data in memory, which is memory consuming (more than 256G memory)
"""
import re
import io
import random
import braceexpand
from utils import *
from decord import VideoReader, cpu
from .base_dataset import BaseDataset, SizedWebDataset
from .base.video_process import read_frames_decord
from .base import inter_video_list_to_dict, get_dataset_size
from .utils import split_data_by_node, shuffle_list
try:
    from azfuse import File
except Exception as e:
    print("azfuse not supported on this cluster, use local file system instead")

class InterleaveVideoTextWDSDataset(BaseDataset):
    def __init__(self, split, data_path, batch_size, tokenizer, image_processor, video_processor, dataset_params, custom_logger):
        super().__init__(split, data_path, batch_size, tokenizer, image_processor, video_processor, dataset_params, custom_logger)
        self.init_wds_dataset(data_path)
    
    def init_wds_dataset(self, data_path):
        self.num_samples, self.num_shards = zip(*(get_dataset_size(sub_data_path.strip(), use_azfuse=self.use_azfuse) for sub_data_path in data_path.split(';')))
        self.num_samples, self.num_shards = sum(self.num_samples), sum(self.num_shards)
        self.custom_logger.info(f"Interleave Video Text Wds Dataset num_samples: {self.num_samples}, num_shards: {self.num_shards}")
        self.custom_logger.info("batch size for Interleave video text wds dataset is: {}".format(self.batch_size))
        urls_train = None
        for sub_data_path in data_path.split(';'):
            if urls_train is None:
                urls_train = list(braceexpand.braceexpand(sub_data_path.strip()))
            else:
                urls_train += list(braceexpand.braceexpand(sub_data_path.strip()))
        # Concatenate the lists of URLs from different datasets
        # Warning: if shuffle size less than batch size, it will sample the same data
        if self.split_data_by_node:
            node_urls = split_data_by_node(urls_train)
            # if data is not enough to split, use all data
            if len(node_urls) == 0:
                node_urls = urls_train
        else:
            node_urls = urls_train
        if self.use_azfuse and self.inter_vwt_tar_pre_cache:
            File.prepare(node_urls[:int(self.inter_vwt_pre_cache_ratio * len(node_urls))])
        # shuffle the dataset
        node_urls = shuffle_list(node_urls, seed=random.randint(0, 1000000))
        self.dataset = (
            SizedWebDataset(node_urls, length=self.num_samples, batch_size=self.batch_size, resampled=True)
            # .shuffle(100)
            .shuffle(300) # shuffle the dataset with buffer, also shards, the time cost is almost the same as 100
            .decode(self.torch_video_with_decord, handler=self.log_and_continue) #  handler=self.log_and_continue
            .to_tuple("mp4", "json")
            .map(self.preprocess_interleaved_video_wds_fn, handler=self.log_and_continue) #  handler=self.log_and_continue
            .batched(self.batch_size)
            .map(inter_video_list_to_dict)
        ).with_epoch(10000)

    def torch_video_with_decord(self, key, data):
        extension = re.sub(r".*[.]", "", key)
        if extension not in "mp4 ogv mjpeg avi mov h264 mpg webm wmv".split():
            return None
        
        # Convert the bytes data to a file-like object
        video_data = io.BytesIO(data)
        vr = VideoReader(video_data, width=256, height=256, num_threads=1, ctx=cpu(0))
        return vr
    
    def __len__(self):
        return self.num_samples