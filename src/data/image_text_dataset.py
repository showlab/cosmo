import webdataset as wds
import braceexpand
import random
from .base_dataset import BaseDataset, SizedWebDataset
from .base import tuple_to_dict, get_dataset_size
from .utils import split_data_by_node, shuffle_list
from utils import *
try:
    from azfuse import File
except Exception as e:
    print("azfuse not supported on this cluster, use local file system instead")

class ImageTextDataset(BaseDataset):
    def __init__(self, split, data_path,  batch_size, tokenizer, image_processor, video_processor, dataset_params, custom_logger):
        assert dataset_params is not None
        super().__init__(split, data_path, batch_size, tokenizer, image_processor, video_processor, dataset_params, custom_logger)
        self.init_wds_dataset(data_path)
    
    def init_wds_dataset(self, data_path):
        """
        data_path can be in a/{00000..00099}.tar,b/{00000..00099}.tar,c/{00000..00099}.tar format
        wds support distributed sampler is not support yet, use resampled=True instead
        see this page for details: https://webdataset.github.io/webdataset/multinode/
        The current distributed sampler may have bugs, see https://github.com/webdataset/webdataset/issues/225
        the larger the shuffle buffer, the more memory it will consume, but the probability of sampling the same data is lower
        # shuffle(n): shuffle the dataset with a buffer of size n; also shuffles shards (see below)
        # see https://webdataset.github.io/webdataset/gettingstarted/
        """
        self.num_samples, self.num_shards = zip(*(get_dataset_size(sub_data_path.strip(), use_azfuse=self.use_azfuse) for sub_data_path in data_path.split(';')))
        self.num_samples, self.num_shards = sum(self.num_samples), sum(self.num_shards)
        self.custom_logger.info(f"{self.split} Image Text Dataset num_samples: {self.num_samples}, num_shards: {self.num_shards}")

        urls_train = None
        for sub_data_path in data_path.split(';'):
            if urls_train is None:
                urls_train = list(braceexpand.braceexpand(sub_data_path.strip()))
            else:
                urls_train += list(braceexpand.braceexpand(sub_data_path.strip()))
        if self.split_data_by_node:
            node_urls = split_data_by_node(urls_train)
            # if data is not enough to split, use all data
            if len(node_urls) == 0:
                node_urls = urls_train
        else:
            node_urls = urls_train
        if self.use_azfuse and self.iwt_tar_pre_cache:
            File.prepare(node_urls[:int(self.iwt_pre_cache_ratio * len(node_urls))])
        # shuffle the dataset 
        node_urls = shuffle_list(node_urls, seed=random.randint(0, 1000000))
        # Concatenate the lists of URLs from different datasets
        # Sometimes the tar file maybe corrupted, we add handler to skip the corrupted tar file
        self.dataset = (
            SizedWebDataset(node_urls, length=self.num_samples, batch_size=self.batch_size, resampled=True, handler=self.log_and_continue)
            .shuffle(300)
            # .shuffle(1000)
            .decode("pil", handler=self.log_and_continue)
            .to_tuple("jpg;png;jpeg", "txt", handler=self.log_and_continue)
            .batched(self.batch_size)
            .map_tuple(self.preprocess_image_fn, self.preprocess_image_text_fn)
            .map(tuple_to_dict)
        ).with_epoch(10000)

    def __len__(self):
        return self.num_samples