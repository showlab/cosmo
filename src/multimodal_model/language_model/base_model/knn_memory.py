import os
import math
import torch
try:
    import faiss
    import faiss.contrib.torch_utils # for gpu
except Exception as e:
    print("faiss not supported on this cluster")
import numpy as np
from pathlib import Path
from functools import wraps

from contextlib import ExitStack, contextmanager
import torch.nn as nn

from einops import rearrange, pack, unpack

# multiprocessing

from joblib import Parallel, delayed, cpu_count

# constants

FAISS_INDEX_GPU_ID = int(os.getenv('FAISS_INDEX_GPU_ID', 0))

DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY = './.tmp/knn.memories'

# helper functions



def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_list(val):
    return val if isinstance(val, list) else [val]

def all_el_unique(arr):
    return len(set(arr)) == len(arr)

@contextmanager
def multi_context(*cms):
    with ExitStack() as stack:
        yield [stack.enter_context(cls) for cls in cms]

def count_intersect(x, y):
    # returns an array that shows how many times an element in x is contained in tensor y
    return np.sum(rearrange(x, 'i -> i 1') == rearrange(y, 'j -> 1 j'), axis = -1)

def check_shape(tensor, pattern, **kwargs):
    return rearrange(tensor, f"{pattern} -> {pattern}", **kwargs)

# a wrapper around faiss IndexIVFFlat
# taking care of expiring old keys automagically
class KNN():
    def __init__(
        self,
        dim,
        max_num_entries,
        cap_num_entries = False,
        M = 15,
        keep_stats = False
    ):
        index_cpu = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        # Move the index to GPU
        self.res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(self.res, 0, index_cpu)
        
        self.max_num_entries = max_num_entries
        self.cap_num_entries = cap_num_entries
        self.is_trained = False
        self.keep_stats = keep_stats

        self.reset()

    def __del__(self):
        if hasattr(self, 'index'):
            del self.index

    def reset(self):
        self.ids = torch.empty((0,), dtype=torch.int32)

        if self.keep_stats:
            self.hits = torch.empty((0,), dtype=torch.int32)
            self.age_num_iterations = torch.empty((0,), dtype=torch.int32)
            self.ages_since_last_hit = torch.empty((0,), dtype=torch.int32)

        self.index.reset()
        self.is_trained = False

    def train(self, x):
        self.index.train(x) 
        self.is_trained = True

    def add(self, x, ids):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if not self.is_trained:
            self.train(x)

        self.ids = torch.cat((torch.tensor(ids, dtype=torch.int32), self.ids))

        if self.keep_stats:
            self.hits = torch.cat((torch.zeros_like(ids), self.hits))
            self.age_num_iterations = torch.cat((torch.zeros_like(ids), self.age_num_iterations))
            self.ages_since_last_hit = torch.cat((torch.zeros_like(ids), self.ages_since_last_hit))

        if self.cap_num_entries and len(self.ids) > self.max_num_entries:
            self.reset()

        return self.index.add(x)  # Convert tensor to numpy before adding


    def search(
        self,
        x,
        topk,
        nprobe=8,
        return_distances=False,
        increment_hits=False,
        increment_age=True
    ):
        if not self.is_trained:
            return torch.full((x.shape[0], topk), -1)

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        distances, indices = self.index.search(x, k=topk)  # Convert tensor to numpy before searching

        # Convert back numpy arrays to tensors
        distances = torch.from_numpy(distances)
        indices = torch.from_numpy(indices)

        if increment_hits and self.keep_stats:
            hits = count_intersect(self.ids, rearrange(indices, '... -> (...)'))
            self.hits += hits

            self.ages_since_last_hit += 1
            self.ages_since_last_hit *= (hits == 0)

        if increment_age and self.keep_stats:
            self.age_num_iterations += 1

        if return_distances:
            return indices, distances

        return indices

# KNN memory layer, where one can store key / value memories
# can automatically take care of a collection of faiss indices (across batch dimension)


class KNNMemory(nn.Module):
    def __init__(
        self,
        dim,
        max_memories=16000,
        dtype=torch.float16
    ):
        super(KNNMemory, self).__init__()
        self.dim = dim
        self.max_memories = max_memories
        self.shape = (max_memories, 2, dim)
        self.db_offset = 0
        self.dtype = dtype
        # self.db = torch.zeros(self.shape, dtype=self.dtype)
        self.register_buffer("db", torch.zeros(self.shape, dtype=self.dtype))
        self.knn = KNN(dim=dim, max_num_entries=max_memories, cap_num_entries=True)

    def clear(self):
        self.knn.reset()
        self.db_offset = 0

    def add(self, memories):
        check_shape(memories, 'b n kv d', d=self.dim, kv=2, b=memories.shape[0])

        memories = memories.reshape(-1, 2, self.dim)
        num_memories = memories.size(0)

        knn_insert_ids = torch.arange(num_memories)

        keys = memories[..., 0, :].contiguous()
        self.knn.add(keys, ids=knn_insert_ids + self.db_offset)

        # add the new memories to the memmap "database"
        add_indices = (torch.arange(num_memories).to(self.db.device) + self.db_offset) % self.max_memories
        self.db[add_indices] = memories

        self.db_offset += num_memories

    def search(
        self,
        queries,
        topk,
        increment_hits = True,
        increment_age = True
    ):
        original_shape = queries.shape
        bsz = original_shape[0]
        check_shape(queries, 'b ... d', d = self.dim, b = original_shape[0])
        queries, ps = pack([queries], 'b * d')
        # print("queries.shape", queries.shape) # 4 x 144 x 64
        # since we use one memory, batchify the queries
        queries = rearrange(queries, 'b n d -> (b n) d')

        fetched_indices = self.knn.search(queries, topk, increment_hits=increment_hits, increment_age=increment_age)

        mask = fetched_indices != -1
        db_indices = torch.where(mask, fetched_indices, torch.zeros_like(fetched_indices))
        all_masks = mask.to(self.db.device)
        key_values = self.db[db_indices % self.max_memories]
        all_key_values = key_values

        all_masks = rearrange(all_masks, '(b n) ... -> b n ...', b=bsz)
        all_key_values = rearrange(all_key_values, '(b n) ... -> b n ...', b=bsz)

        all_key_values = all_key_values.masked_fill(~rearrange(all_masks, '... -> ... 1 1'), 0.)
        all_key_values, = unpack(all_key_values, ps, 'b * n kv d')
        all_masks, = unpack(all_masks, ps, 'b * n')

        return all_key_values, all_masks

    def __del__(self):
        pass