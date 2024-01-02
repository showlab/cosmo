"""Evaluate on image-text retrieval datasets."""

import os

import datasets
import open_clip
import torch
from clip_benchmark.datasets.builder import image_captions_collate_fn
from src.eval.eval_tasks.utils.datacomp import zeroshot_retrieval as zsr


class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        super().__init__()
        self._dataset = hf_dataset
        self.transform = (lambda x: x) if transform is None else transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int):
        return (
            self.transform(self._dataset[index]["image"]),
            self._dataset[index]["caption"],
        )


def evaluate_retrieval_dataset(
    task, eval_model, data_root=None, batch_size=64, num_workers=4
):
    """Evaluate CLIP model on retrieval task."""

    model = eval_model
    transform, tokenizer, device = eval_model.image_processor, eval_model.tokenizer, eval_model.device

    data_abs_root = os.path.join(data_root, "hf_cache")
    dataset = RetrievalDataset(
        datasets.load_dataset(
            f"{data_abs_root}/{task.replace('retrieval/', '')}",
            split="test",
            # f"nlphuji/{task.replace('retrieval/', '')}",
            # split="test",
            # cache_dir=os.path.join(data_root, "hf_cache")
            # if data_root is not None
            # else None,
        ),
        transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=image_captions_collate_fn,
    )

    metrics = zsr.evaluate(
        model, dataloader, tokenizer, recall_k_list=[1, 5, 10], device=device
    )
    metrics["mean_recall@1"] = 0.5 * (
        metrics["text_retrieval_recall@1"] + metrics["image_retrieval_recall@1"]
    )
    return metrics
