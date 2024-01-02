import os
from tqdm import tqdm
import json
import more_itertools
import uuid
from collections import defaultdict
import torch
from src.eval.models import eval_base_model
from src.eval.data.retrieval_dataset import RetrievalDataset
from src.eval.eval_tasks.util import prepare_eval_samples, get_query_set, sample_batch_demos_from_query_set
from src.eval.eval_tasks.utils.retrieval_metric import t2v_metrics, v2t_metrics, sim_matrix


def evaluate_imagenet(
    config: dict,
    eval_model: eval_base_model.BaseEvalModel,
    seed: int = 42,
    metric_fns: list = [t2v_metrics, v2t_metrics],
    dataset_name: str = "coco",
    num_shots: int = 8,
):
    """Evaluate a model on COCO dataset.

    Args:
        args (argparse.Namespace): arguments
        eval_model (eval_model.BaseEvalModel): model to evaluate
        seed (int, optional): seed for random number generator. Defaults to 42.
        dataset_name (str, optional): dataset to evaluate on. Can be "coco" or "flickr". Defaults to "coco".
    Returns:
        float: CIDEr score

    """
    num_samples = config['general']['num_samples']
    if num_shots <= 8:
        batch_size = config['general']['batch_size']
    else:
        batch_size = 4

    if dataset_name == "imagenet":
        image_dir_path = os.path.join(config['general']['data_root'], config['datasets']['coco']['ret_image_dir_path'])
        val_annotations_json_path = os.path.join(config['general']['data_root'], config['datasets']['coco']['ret_val_annotations_json_path'])
        test_annotations_json_path = os.path.join(config['general']['data_root'], config['datasets']['coco']['ret_test_annotations_json_path'])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # for retrieval task, we only need to evaluate on test set and no query set is needed
    test_dataset = RetrievalDataset(
        image_dir_path=image_dir_path,
        annotations_path=test_annotations_json_path,
        is_train=False,
        dataset_name=dataset_name,
    )

    test_dataset = prepare_eval_samples(
        test_dataset,
        num_samples if num_samples > 0 else len(test_dataset),
        seed,
    )

    text_embed_list = []
    visual_embed_list = []

    # modify the following code, for text, add classification prompt
    # classification_prompt
    for batch in more_itertools.chunked(
        tqdm(test_dataset, desc=f"Running retrieval inference {dataset_name.upper()}"),
        batch_size,
    ):
        batch_images = []
        batch_text = []
        for i in range(len(batch)):
            batch_images.append([batch[i]["image"]])
            batch_text.append(batch[i]["caption"])
        text_embeds, image_embeds = eval_model.get_embeddings(
            batch_images=batch_images,
            batch_text=batch_text,
        )
        text_embed_list.append(text_embeds.float().cpu().detach())
        visual_embed_list.append(image_embeds.float().cpu().detach())

    text_embeds_mat = torch.cat(text_embed_list, dim=0)
    visual_embeds_mat = torch.cat(visual_embed_list, dim=0)

    sims = sim_matrix(visual_embeds_mat, text_embeds_mat).numpy()
    nested_metrics = {}
    # print(metric_fns)
    for metric in metric_fns:
        metric_name = metric.__name__
        res = metric(sims)
        nested_metrics[metric_name] = res
    return nested_metrics