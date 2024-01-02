import torch
from itertools import cycle
from torch.utils.data.dataset import ConcatDataset
from .image_text_dataset import ImageTextDataset
from .interleave_image_text_dataset import InterleaveImageTextDataset
from .video_text_dataset import VideoTextDataset
from .interleave_video_text_dataset import InterleaveVideoTextDataset

# # Usage, if multiple dataset, 
# # urls = list(braceexpand(dataset1)) + list(braceexpand(dataset2))
# # dataset.= webdataset.Dataset(urls) 
# # look https://github.com/webdataset/webdataset/issues/19 for example

def dataset_func(split, batch_size, type_name, data_path, config, custom_logger):
    """
    iwt and itwt are wdb
    image text: image with shape (B, 1, 1, 3, 224, 224), text token with shape (B, 32)
    interlevel image text: image with shape (B, 3, 1, 1, 224, 224), text token with shape (B, 256), text is 8 times larger than image_text pairs
    video text: video with shape (B, 1, 1, 3, 224, 224), text token with shape (B, 32)
    interlevel data requires more GPU memory, so we need to reduce the batch size    
    """
    if type_name == "img_txt":
        real_batch_size = batch_size * 2
    elif type_name in ["inter_img_txt", "inter_vid_txt"]:
        real_batch_size = max(4, batch_size//2)
    else:
        real_batch_size = batch_size
    if type_name == "img_txt":
        dataset = ImageTextDataset(split=split, data_path=data_path,  batch_size=real_batch_size, **config)
    elif type_name == "inter_img_txt":
        dataset = InterleaveImageTextDataset(split=split, data_path=data_path, batch_size=real_batch_size, **config)
    elif type_name == "vid_txt":
        dataset = VideoTextDataset(split=split, data_path=data_path, batch_size=None, **config)
    elif type_name == "inter_vid_txt":
        dataset = InterleaveVideoTextDataset(split=split, data_path=data_path, batch_size=None, **config)
    else:
        raise ValueError(f"Unknown type_name: {type_name}")
    custom_logger.info("Data type: {}  Split: {} Batch size: {}".format(type_name, split, real_batch_size))
    if type_name in ["img_txt", "inter_img_txt"]:
        return dataset.dataset
    else:
        return dataset

def multi_modality_dataset(batch_size, num_workers, data_paths, split, config, custom_logger):
    datasets = []
    for key, path in data_paths.items():
        if path:
            type_name = key.split("_path")[0]
            dataset = dataset_func(split, batch_size, type_name, path, config, custom_logger)
            datasets.append(dataset)
    return ConcatDataset(datasets)