import os
import torch
import math
import random
import pandas as pd

def identity(x):
    return x

def filter_no_caption_or_no_image(sample):
    return ("txt" in sample) and (
        "png" in sample or "jpg" in sample or "jpeg" in sample
    )

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    if "No images in sample" in str(exn) or "Only one image in sample" in str(
        exn
    ):  # Avoid spamming logs with these
        return True
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True



# this implementation is from webdataset https://github.com/webdataset/webdataset/blob/cfd0474e7e332e0b65129890c9b6a5f1d78340d6/webdataset/utils.py#L91
def pytorch_worker_info(group=None):
    """Return node and worker info for PyTorch and some distributed environments."""
    rank = 0
    world_size = 1
    worker = 0
    num_workers = 1
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        try:
            import torch.distributed

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                group = group or torch.distributed.group.WORLD
                rank = torch.distributed.get_rank(group=group)
                world_size = torch.distributed.get_world_size(group=group)
        except ModuleNotFoundError:
            pass
    if "WORKER" in os.environ and "NUM_WORKERS" in os.environ:
        worker = int(os.environ["WORKER"])
        num_workers = int(os.environ["NUM_WORKERS"])
    else:
        try:
            import torch.utils.data

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                worker = worker_info.id
                num_workers = worker_info.num_workers
        except ModuleNotFoundError:
            pass

    return rank, world_size, worker, num_workers


"""
split data into different nodes with webdataset tar files
"""

def split_data_by_node(urls, strategy="interleaved"):
    """
    Distribute URLs across nodes based on the given distribution strategy.

    
    Parameters:
    - urls (list): A list of URLs that need to be distributed.
    - strategy (str): The distribution strategy. It can take one of the following values:
        * "chunk": Divide the URLs into contiguous chunks and assign to each node. Very fast with azcopy.
        * "interleaved": Distribute the URLs in an interleaved manner across nodes. Very fast with azcopy.
        * "shuffled_chunk": Shuffle the URLs and then divide into contiguous chunks.
    
    Returns:
    - list: A subset of URLs that are assigned to the current node.
    
    Example:
        Given 16 URLs and 4 nodes:
        
        "chunk" strategy:
            Node 0 gets URLs 0-3
            Node 1 gets URLs 4-7
            Node 2 gets URLs 8-11
            Node 3 gets URLs 12-15
            
        "interleaved" strategy:
            Node 0 gets URLs 0, 4, 8, 12
            Node 1 gets URLs 1, 5, 9, 13
            Node 2 gets URLs 2, 6, 10, 14
            Node 3 gets URLs 3, 7, 11, 15
        
        "shuffled_chunk" strategy (depends on shuffle):
            e.g., Shuffle gives [5, 11, 1, 13, 9, 7, 2, 8, 0, 14, 3, 12, 4, 10, 6, 15]
            Node 0 gets URLs 5, 11, 1, 13
            Node 1 gets URLs 9, 7, 2, 8
            Node 2 gets URLs 0, 14, 3, 12
            Node 3 gets URLs 4, 10, 6, 15
    """
    print('*'*80)
    print("split_data_by_node ing..................")
    gpus_per_node = torch.cuda.device_count()
    # gpus_per_node = 1 # for local debug
    rank, world_size, worker, num_workers = pytorch_worker_info()
    print("rank: {}, world_size: {}, worker: {}, num_workers: {}, gpus_per_node: {}".format(rank, world_size, worker, num_workers, gpus_per_node))
    node_rank = rank // gpus_per_node
    node_world_size = world_size // gpus_per_node

    if strategy == "chunk":
        urls_per_node = math.ceil(len(urls) / node_world_size)
        start_idx = node_rank * urls_per_node
        end_idx = min(start_idx + urls_per_node, len(urls))
        node_urls = urls[start_idx:end_idx]
    
    elif strategy == "interleaved":
        node_urls = urls[node_rank::node_world_size]
    
    elif strategy == "shuffled_chunk":
        shuffled_urls = random.sample(urls, len(urls))
        urls_per_node = math.ceil(len(shuffled_urls) / node_world_size)
        start_idx = node_rank * urls_per_node
        end_idx = min(start_idx + urls_per_node, len(urls))
        node_urls = shuffled_urls[start_idx:end_idx]
    
    else:
        raise ValueError(f"Unknown strategy {strategy}")
    
    print(f"Node {node_rank} has {len(node_urls)} URLs of {len(urls)} total.")
    print('*'*80)
    return node_urls


def split_json_data_by_node(json_data, strategy="interleaved"):
    """
    Distribute JSON data across nodes based on the given distribution strategy.

    Parameters:
    - json_data (list or pd.DataFrame): JSON data to be distributed.
    - strategy (str): The distribution strategy. Options: "chunk", "interleaved", "shuffled_chunk".
    - node_rank (int): Rank of the current node.
    - node_world_size (int): Total number of nodes.

    Returns:
    - list or pd.DataFrame: Subset of JSON data assigned to the current node.
    """
    print('*'*80)
    print("split_data_by_node ing..................")
    gpus_per_node = torch.cuda.device_count()
    # gpus_per_node = 1 # for local debug
    rank, world_size, worker, num_workers = pytorch_worker_info()
    print("rank: {}, world_size: {}, worker: {}, num_workers: {}, gpus_per_node: {}".format(rank, world_size, worker, num_workers, gpus_per_node))
    node_rank = rank // gpus_per_node
    node_world_size = world_size // gpus_per_node

    if isinstance(json_data, pd.DataFrame):
        data_length = len(json_data)
    else:
        data_length = len(json_data)  # assuming json_data is a list

    if strategy == "chunk":
        data_per_node = data_length // node_world_size
        start_idx = node_rank * data_per_node
        end_idx = (node_rank + 1) * data_per_node if node_rank < node_world_size - 1 else data_length
        node_data = json_data[start_idx:end_idx]
    elif strategy == "interleaved":
        node_data = json_data[node_rank::node_world_size]
    elif strategy == "shuffled_chunk":
        random.shuffle(json_data)
        return split_json_data_by_node(json_data, strategy="chunk", node_rank=node_rank, node_world_size=node_world_size)
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    print(f"Node {node_rank} has {len(node_data)} items of {data_length} total.")
    return node_data


def shuffle_list(urls, seed=1234):
    """
    shuffle the list of urls
    """
    random.seed(seed)
    random.shuffle(urls)
    return urls