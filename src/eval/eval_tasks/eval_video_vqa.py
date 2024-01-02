import os
from src.eval.data.video_vqa_dataset import VIDEOVQADataset
from src.eval.models import eval_base_model
from src.eval.eval_tasks.utils.video_vqa_metric import compute_video_vqa_accuracy, postprocess_video_vqa_generation
from src.eval.eval_tasks.utils.ok_vqa_utils import postprocess_ok_vqa_generation
import more_itertools
from src.eval.eval_tasks.util import *
from tqdm import tqdm
import json
import uuid

def evaluate_video_vqa(
    config: dict,
    eval_model: eval_base_model.BaseEvalModel,
    seed: int = 42,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = -2.0,
    num_shots: int = 8,
    dataset_name: str = "msvd_qa",
    split: str = "test",
):
    """
    ...
    Args:
        config (dict): Configuration dictionary.
        ...
        dataset_name (string): Type of VQA dataset
    Returns:
        float: Accuracy score
    """
    if num_shots <= 4:
        batch_size = 2
    else:
        batch_size =1
    num_samples = config['general']['num_samples']
    query_set_size = config['general']['query_set_size']

    # Get dataset configuration
    dataset_config = config['datasets'].get(dataset_name)
    if dataset_config is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    video_dir_path = os.path.join(config['general']['data_root'], dataset_config['video_dir_path'])
    annotations_path = os.path.join(config['general']['data_root'], dataset_config['annotations_path'])
    # annotations_path
    if split == "test":
        test_annotations_json_path = os.path.join(annotations_path, dataset_config['test_annotations_json_path'])
    else:
        test_annotations_json_path = os.path.join(annotations_path, dataset_config['val_annotations_json_path'])

    train_dataset = VIDEOVQADataset(
        video_dir_path,
        annotations_path,
        split="train",
        dataset_name=dataset_name,
    )

    test_dataset = VIDEOVQADataset(
        video_dir_path,
        annotations_path,
        split="test" if split == "test" else "val",
        dataset_name=dataset_name,
    )

    # effective_num_shots = num_shots if num_shots > 0 else 2
    effective_num_shots = num_shots

    test_dataset = prepare_eval_samples(
        test_dataset,
        num_samples if num_samples > 0 else len(test_dataset),
        seed,
    )

    query_set_size = min(query_set_size, len(train_dataset)) # prevent query set size from being larger than the training set

    in_context_samples = get_query_set(train_dataset, query_set_size, seed)
    predictions = []

    for batch in more_itertools.chunked(
        tqdm(test_dataset, desc=f"Running vqa inference {dataset_name.upper()} shots={num_shots}"),
        batch_size,
    ):
        batch_demo_samples = sample_batch_demos_from_query_set(
            in_context_samples, effective_num_shots, len(batch)
        )

        batch_videos = []
        batch_text = []
        for i in range(len(batch)):
            if num_shots > 0:
                context_videos = [x["video"] for x in batch_demo_samples[i]]
            else:
                context_videos = []
            batch_videos.append(context_videos + [batch[i]["video"]])

            context_text = "".join(
                [
                    eval_model.vqa_prompt(
                        question=x["question"], answer=x["answers"][0]
                    )
                    for x in batch_demo_samples[i]
                ]
            )

            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<visual>", "")

            batch_text.append(
                context_text + eval_model.vqa_prompt(question=batch[i]["question"])
            )
        with torch.no_grad():
            outputs = eval_model.get_video_outputs(
                batch_videos=batch_videos,
                batch_text=batch_text,
                max_generation_length=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )

        process_function = (
            postprocess_video_vqa_generation
        )

        new_predictions = map(process_function, outputs)
        predictions.extend(
            [
                {"answer": p, "question_id": int(sample["question_id"])}
                for p, sample in zip(new_predictions, batch)
            ]
        )
        # print(batch_text[-1])
        # print(predictions[-1])
    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    with open(f"{dataset_name}_results_{random_uuid}.json", "w") as f:
        f.write(json.dumps(predictions, indent=4))


    acc = compute_video_vqa_accuracy(
        f"{dataset_name}_results_{random_uuid}.json",
        test_annotations_json_path,
    )

    # delete the temporary file
    os.remove(f"{dataset_name}_results_{random_uuid}.json")
    return acc
