import torch.distributed as dist
import os
from itertools import cycle
from torch.utils.data import DataLoader, DistributedSampler
from .image_text_dataset import ImageTextDataset
from .interleave_image_text_dataset import InterleaveImageTextDataset
from .video_text_dataset import VideoTextDataset
from .video_text_dataset_wds import VideoTextWDSDataset
from .interleave_video_text_dataset import InterleaveVideoTextDataset
from .interlevel_video_text_dataset_wds import InterleaveVideoTextWDSDataset

class DataLoaderWithType(DataLoader):
    def __init__(self, dataset, type_name, data_type='wds', sampler=None, **kwargs):
        super().__init__(dataset, sampler=sampler, **kwargs)
        self.type_name = type_name
        self.data_type = data_type
        self.dist_sampler = sampler
        self.dist_sampler_current_epoch = 0
        if self.dist_sampler is not None:
            print("set dist sampler for type {self.type_name}")
    
    def increase_dist_sampler_epoch(self, stop_type='data_iter_stop'):
        """
        stop_type in: data_iter_stop, epoch_stop
        """
        if dist.get_rank()==0 and self.dist_sampler is not None:
            self.dist_sampler_current_epoch += 1
            self.dist_sampler.set_epoch(self.dist_sampler_current_epoch)
            print(f"Set epoch for {self.type_name} distributed resampler to {int(self.dist_sampler_current_epoch)} with reason {stop_type}")


# v1: iteratively sample from each dataloader, however, the dataloader is not shuffled, so the training is not stable
class MultiModalDataLoader:
    def __init__(self, dataloaders, split, custom_logger, sampling_strategy='round_robin'):
        self.dataloaders = dataloaders
        self.iterators = [(dataloader.type_name, iter(dataloader)) for dataloader in dataloaders]
        self.dataloader_cycle = cycle(range(len(dataloaders)))
        self.split = split
        self.sample_strategy = sampling_strategy
        self.custom_logger = custom_logger
        dataloader_lens = [len(dataloader) for dataloader in dataloaders]
        self.custom_logger.info(f"Split: {self.split} Dataloader type and length: {list(zip([dataloader.type_name for dataloader in dataloaders], dataloader_lens))}")
        # Keep track of the last iteration for each dataloader
        self.last_iteration = [False] * len(dataloaders)

    def __len__(self):
        if self.sample_strategy == "round_robin":
            return sum([len(dataloader) for dataloader in self.dataloaders])
        elif self.sample_strategy == "min":
            return min([len(dataloader) for dataloader in self.dataloaders]) * len(self.dataloaders)
        elif self.sample_strategy == "max":
            return max([len(dataloader) for dataloader in self.dataloaders]) * len(self.dataloaders)
        else:
            raise NotImplementedError(f"Sampling strategy {self.sample_strategy} is not implemented")

    def __iter__(self):
        return self

    def __next__(self):
        """
        for dataset based on pytorch Dataset, we need to write the custom distributed sampler; 
        (prevent contrastive loss meet same image and text pair)
        """
        dataloader_index = next(self.dataloader_cycle)
        type_name, iterator = self.iterators[dataloader_index]
        try:
            batch = next(iterator)
        except StopIteration:
            self.iterators[dataloader_index] = (type_name, iter(self.dataloaders[dataloader_index]))
            if self.sample_strategy in ["round_robin", "max"]:
                self.dataloaders[dataloader_index].increase_dist_sampler_epoch(stop_type='data_iter_stop')
            type_name, iterator = self.iterators[dataloader_index]
            batch = next(iterator)
        data = {key: batch[key] for key in ['visual', 'input_ids', 'attention_mask', 'labels']}
        data['type_name'] = type_name
        data['data_type'] = self.dataloaders[dataloader_index].data_type
        return data  # now we return both the data and the type_name


# # Usage, if multiple dataset, 
# # urls = list(braceexpand(dataset1)) + list(braceexpand(dataset2))
# # dataset.= webdataset.Dataset(urls) 
# # look https://github.com/webdataset/webdataset/issues/19 for example

def dataloader_func(split, batch_size, type_name, num_workers, data_path, config, custom_logger):
    """
    iwt and itwt are wdb with IterableDataset (Can Use DistributedSampler Directly)
    others are based on pytorch Dataset, (We Need To Write the CustomDistributedSampler)
    image text: image with shape (B, 1, 1, 3, 224, 224), text token with shape (B, 32)
    interlevel image text: image with shape (B, 3, 1, 1, 224, 224), text token with shape (B, 256), text is 8 times larger than image_text pairs
    video text: video with shape (B, 1, 1, 3, 224, 224), text token with shape (B, 32)
    interlevel data requires more GPU memory, so we need to reduce the batch size    
    """
    print(f"Create dataloader for {type_name}")
    if type_name == "img_txt":
        real_batch_size = int(batch_size * 2)
    elif type_name in ["inter_img_txt", "inter_vid_txt_wds", "inter_vid_txt_tsv", "vid_txt_wds", "vid_txt_tsv"]:
        real_batch_size = max(4, batch_size//2)
    else:
        real_batch_size = batch_size
    # real_batch_size = batch_size
    if type_name == "img_txt":
        dataset = ImageTextDataset(split=split, data_path=data_path,  batch_size=real_batch_size, **config)
    elif type_name == "inter_img_txt":
        dataset = InterleaveImageTextDataset(split=split, data_path=data_path, batch_size=real_batch_size, **config)
    elif type_name == "vid_txt_wds":
        dataset = VideoTextWDSDataset(split=split, data_path=data_path, batch_size=real_batch_size, **config)
    elif type_name == "vid_txt_tsv":
        dataset = VideoTextDataset(split=split, data_path=data_path, batch_size=None, **config)
        sampler = DistributedSampler(dataset) if split == "train" else None
    elif type_name == "inter_vid_txt_tsv":
        dataset = InterleaveVideoTextDataset(split=split, data_path=data_path, batch_size=None, **config)
        sampler = DistributedSampler(dataset) if split == "train" else None
    elif type_name == "inter_vid_txt_wds":
        dataset = InterleaveVideoTextWDSDataset(split=split, data_path=data_path, batch_size=real_batch_size, **config)
    else:
        raise ValueError(f"Unknown type_name: {type_name}")
    data_type = "tsv" if "tsv" in type_name else "wds" # tsv is default
    custom_logger.info("Data type: {}  Split: {} Batch size: {}".format(type_name, split, real_batch_size))
    if type_name in ["img_txt", "inter_img_txt", "vid_txt_wds", "inter_vid_txt_wds"]:
        dataloader = DataLoaderWithType(dataset.dataset, type_name, data_type=data_type, num_workers=num_workers, batch_size=None)
    else:
        dataloader = DataLoaderWithType(dataset, type_name, data_type=data_type, sampler=sampler, num_workers=num_workers, batch_size=real_batch_size)
    return dataloader

def is_loader_required(type_name, video_read_mode, inter_video_read_mode, dataset_params):
    """
    Video have tsv/wds format, when both read model and data are in same format;
    Control data type with use.
    """
    print('*'*50)
    print(f"type_name: {type_name}, data_read_mode: {video_read_mode}")
    if (video_read_mode == "tsv" and "vid_txt_wds" == type_name) or \
        (video_read_mode == "wds" and "vid_txt_tsv" == type_name) or \
            (inter_video_read_mode == "tsv" and "inter_vid_txt_wds" == type_name) or \
                (inter_video_read_mode == "wds" and "inter_vid_txt_tsv" == type_name):
        return False
    
    relevant_params = ["img_txt", "inter_img_txt", "vid_txt_wds", "vid_txt_tsv", "inter_vid_txt_wds", "inter_vid_txt_tsv"]
    for param in relevant_params:
        if param == type_name:
            if param.endswith("tsv") or param.endswith("wds"): 
                if not dataset_params.get(param[:-4], {}).get(f"use_{param[:-4]}", True):
                    return False
            else:
                if not dataset_params.get(param, {}).get(f"use_{param}", True):
                    return False
            
    return True

def add_prefix_to_paths(prefix, rel_path):
    paths = rel_path.split(';')
    abs_paths = [os.path.join(prefix, path.strip().lstrip('/')) for path in paths]
    return '; '.join(abs_paths)

def multi_modality_dataloader(batch_size, num_workers, data_paths, video_read_mode, inter_video_read_mode, prefix, split, config, custom_logger, dataset_params):
    dataloaders = []
    for key, rel_path in data_paths.items():
        if rel_path:
            type_name = key.split("_path")[0]
            if not is_loader_required(type_name, video_read_mode, inter_video_read_mode, dataset_params):
                print(f"Skip type {type_name} for {split}")
                continue
            abs_path = add_prefix_to_paths(prefix, rel_path)
            dataloader = dataloader_func(split, batch_size, type_name, num_workers, abs_path, config, custom_logger)
            dataloaders.append(dataloader)
    return MultiModalDataLoader(dataloaders, split=split, custom_logger=custom_logger, sampling_strategy=config['dataset_params']['sampling_strategy'])