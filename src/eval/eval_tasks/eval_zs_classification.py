import os
import yaml
import json

from src.eval.eval_tasks.utils.datacomp.fairness_eval import (
    evaluate_dollar_street_dataset,
    evaluate_fairface_dataset,
    evaluate_geode_dataset,
)
from src.eval.eval_tasks.utils.datacomp.retr_eval import evaluate_retrieval_dataset
from src.eval.eval_tasks.utils.datacomp.wds_eval import evaluate_webdataset
from src.eval.eval_tasks.utils.datacomp.wilds_eval import evaluate_wilds_dataset
from src.eval.eval_tasks.utils.datacomp.wino_eval import evaluate_winogavil_dataset
from src.eval.models import eval_base_model

# evaluate_model(task_key, train_info, data_root, dataset_size, batch_size=64):
def evaluate_zs_classification(config: dict, eval_model: eval_base_model.BaseEvalModel, dataset_name: str = "datacomp", **kwargs):
    """
    evaluate on all tasks in datacomp
    """
    batch_size = config['general']['batch_size']
    dataset_config = config['datasets'].get(dataset_name)
    if dataset_config is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    data_root = config['general']['data_root'] +  dataset_config['image_dir_path']
    dataset_size = config['general']['num_samples']
    # if dataset_size == -1:
    #     dataset_size = None

    dataset_size = None
    # Get list of datasets, generally 38 datasets
    with open(os.path.join(os.path.dirname(__file__), "utils/datacomp/tasklist.yml")) as f:
        tasks = yaml.safe_load(f)
    # Iterate through the datasets and record the metrics
    metrics_dict = {}
    for task_key in tasks:
        task_name = tasks[task_key].get("name", task_key) 
        print(f"Evaluating datacomp {task_name}...\n")
        if task_key.startswith("retrieval/"):
            metrics = evaluate_retrieval_dataset(
                task_key,
                eval_model,
                data_root=data_root,
                batch_size=batch_size,
            )
        elif task_key.startswith("wilds/"):
            metrics = evaluate_wilds_dataset(
                task_key,
                eval_model,
                data_root=data_root,
                dataset_len=dataset_size,
                batch_size=batch_size,
            )
        elif task_key.startswith("fairness/"):
            eval_fn = {
                "fairness/dollar_street": evaluate_dollar_street_dataset,
                "fairness/geode": evaluate_geode_dataset,
                "fairness/fairface": evaluate_fairface_dataset,
                "fairness/utkface": evaluate_fairface_dataset,
            }.get(task_key)
            if eval_fn is not None:
                metrics = eval_fn(
                    task_key,
                    eval_model,
                    data_root=data_root,
                    dataset_len=dataset_size,
                    batch_size=batch_size,
                )
            else:
                metrics = {}
        elif task_key.startswith("misc/"):
            if task_key == "misc/winogavil":
                metrics = evaluate_winogavil_dataset(
                    eval_model,
                    data_root=data_root,
                    batch_size=batch_size,
                )
            else:
                metrics = {}
        else:
            metrics = evaluate_webdataset(
                task_key,
                eval_model,
                data_root=data_root,
                dataset_len=dataset_size,
                batch_size=batch_size,
            )
        metrics_dict[task_name] = metrics
        print(f"Metrics for {task_name}: {metrics}\n")
    # write metrics to json file
    with open(os.path.join(os.path.dirname(__file__), "utils/datacomp/metrics.json"), "w") as f:
        json.dump(metrics_dict, f)
    return metrics
