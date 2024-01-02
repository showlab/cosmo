import numpy as np
from tqdm import tqdm
import torch
import math
import more_itertools
import random
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import os

from src.eval.data.classification_dataset import ClassificationDataset
from src.eval.eval_tasks.utils.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL, openai_imagenet_classnames, find_sub_list, HM_CLASSNAMES
from src.eval.eval_tasks.util import prepare_eval_samples, get_query_set, sample_batch_demos_from_query_set, get_predicted_classnames
from src.eval.eval_tasks.utils.rices import RICES


# do classification by zero-shot matching to the class names, it include three steps:
# 1. read all the class names from the dataset and prompt, and get the class name embedding from the model
# 2. get the image embedding from the model
# 3. calculate the similarity between the class name embedding and the image embedding, and get the top-k class names

def evaluate_classification(
    config: dict,
    eval_model,
    seed: int = 42,
    num_shots: int = 8,
    dataset_name: str = "hatefulmemes",
    rices: bool = False,
    cached_features=None,
    no_kv_caching=True,
    use_prompt_ensembling: bool = False,
    rices_vision_encoder_path: str = "ViT-L-14",
    rices_vision_encoder_pretrained: str = "openai",
):
    """
    Evaluate a model on ImageNet dataset.

    Args:
        eval_model (eval_model.BaseEvalModel): model to evaluate
        batch_size (int): batch size
        seed (int, optional): random seed. Defaults to 42.
        num_samples (int, optional): number of samples to evaluate on. Defaults to 5000 samples.
        num_shots (int, optional): number of shots to use. Defaults to 8.

    Returns:
        float: accuracy score
    """
    num_samples = config['general']['num_samples']
    query_set_size = config['general']['query_set_size']
    if num_shots <= 8:
        batch_size = config['general']['batch_size']
    else:
        batch_size = 4

    if dataset_name == "hatefulmemes":
        image_dir_path = os.path.join(config['general']['data_root'], config['datasets']['hatefulmemes']['image_dir_path'])
        train_annotations_json_path = os.path.join(config['general']['data_root'], config['datasets']['hatefulmemes']['train_annotations_json_path'])
        test_annotations_json_path = os.path.join(config['general']['data_root'], config['datasets']['hatefulmemes']['test_annotations_json_path'])
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")
    
    if dataset_name == "hatefulmemes":
        train_dataset = ClassificationDataset(
            image_dir_path,
            train_annotations_json_path,
        )
        test_dataset = ClassificationDataset(
            image_dir_path,
            test_annotations_json_path,
        )
        prompt_fn = lambda x: eval_model.get_hateful_memes_prompt(
            text=x["ocr"], label=x["class_name"]
        )
        all_class_names = HM_CLASSNAMES
        k = 1
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    class_id_to_name = dict(zip(range(len(all_class_names)), all_class_names))

    effective_num_shots = num_shots if num_shots > 0 else 2


    test_dataloader = prepare_eval_samples(
        test_dataset,
        num_samples if num_samples > 0 else len(test_dataset),
        batch_size,
    )

    if rices:
        rices_dataset = RICES(
            train_dataset,
            eval_model.device,
            batch_size,
            cached_features=cached_features,
            vision_encoder_path=rices_vision_encoder_path,
            vision_encoder_pretrained=rices_vision_encoder_pretrained,
        )
    else:
        # subset of the training set to sample context images from
        query_set = get_query_set(train_dataset, query_set_size, seed)


    predictions = []
    for batch in more_itertools.chunked(
        tqdm(test_dataloader, desc=f"Running classification inference {dataset_name.upper()}"),
        batch_size,
    ):
        if rices:
            batch_demo_samples = rices_dataset.find(batch, effective_num_shots)
        else:
            batch_demo_samples = sample_batch_demos_from_query_set(
                query_set, effective_num_shots, len(batch)
            )

        # set up prompt ensembling
        num_permutations = (
            min(6, math.factorial(effective_num_shots)) if use_prompt_ensembling else 1
        )
        logprobs = []
        for _ in range(num_permutations):
            batch_images, batch_text = [], []
            for i in range(len(batch)):
                if use_prompt_ensembling:
                    random.shuffle(batch_demo_samples[i])

                if effective_num_shots > 0:
                    context_images = [x["image"] for x in batch_demo_samples[i]]
                else:
                    context_images = []
                batch_images.append(context_images + [batch[i]["image"]])

                context_text = "".join([prompt_fn(x) for x in batch_demo_samples[i]])

                # Keep the text but remove the image tags for the zero-shot case
                if num_shots == 0:
                    context_text = context_text.replace("<image>", "")

                batch_text.append(
                    context_text
                    + prompt_fn({"ocr": batch[i]["ocr"], "class_name": None})
                )

            # get predicted class names
            logprobs.append(
                eval_model.get_rank_classifications(
                    batch_text,
                    batch_images,
                    all_class_names,
                    use_cache=(not no_kv_caching),
                    normalize_length=True,
                )
            )

        # ensemble logprobs together
        logprobs = torch.mean(torch.stack(logprobs, dim=-1), dim=-1)
        # RuntimeError: "topk_cpu" not implemented for 'Half'
        predicted_classnames, predicted_logprobs = get_predicted_classnames(
            logprobs.float(),
            k,
            class_id_to_name,
        )

        # compute accuracy
        for i, topk in enumerate(predicted_classnames):
            y_i = batch[i]["class_name"]
            score = torch.exp(
                predicted_logprobs.float()[i][0] - torch.logsumexp(logprobs.float()[i], dim=0)
            ).item()
            predictions.append(
                {
                    "id": batch[i]["id"],
                    "gt_label": y_i,
                    "pred_label": topk[0],
                    "pred_score": score,
                }
            )

    # all gather
    all_predictions = predictions

    # all_predictions = [
    #     item for sublist in all_predictions for item in sublist
    # ]  # flatten

    if dataset_name == "hatefulmemes":
        # return ROC-AUC score
        greater_label = max(all_class_names)
        gts = [pred["gt_label"] for pred in all_predictions]
        pred_scores = [
            pred["pred_score"]
            if pred["pred_label"] == greater_label
            else 1 - pred["pred_score"]
            for pred in all_predictions
        ]
        return roc_auc_score(gts, pred_scores)
    else:
        raise   ValueError(f"Unsupported dataset {dataset_name}")
