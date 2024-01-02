from src.eval.models import eval_base_model
from src.eval.data.video_captioning_dataset import VideoCaptionDataset
import os
from src.eval.eval_tasks.util import prepare_eval_samples, get_query_set, sample_batch_demos_from_query_set
from tqdm import tqdm
import json
import more_itertools
import uuid
from collections import defaultdict
from src.eval.eval_tasks.utils.coco_metric import compute_cider, postprocess_captioning_generation

def evaluate_video_captioning(
    config: dict,
    eval_model: eval_base_model.BaseEvalModel,
    seed: int = 42,
    max_generation_length: int = 20,
    num_beams: int = 3,
    length_penalty: float = -2.0,
    num_shots: int = 8,
    dataset_name: str = "coco",
    split: str = "test",
):
    """Evaluate a model on COCO dataset.

    Args:
        args (argparse.Namespace): arguments
        eval_model (eval_model.BaseEvalModel): model to evaluate
        seed (int, optional): seed for random number generator. Defaults to 42.
        max_generation_length (int, optional): maximum length of the generated caption. Defaults to 20.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of in-context samples to use. Defaults to 8.
        dataset_name (str, optional): dataset to evaluate on. Can be "coco" or "flickr". Defaults to "coco".
    Returns:
        float: CIDEr score

    """
    num_samples = config['general']['num_samples']
    query_set_size = config['general']['query_set_size']
    if num_shots <= 4:
        batch_size = 2
    else:
        batch_size =1

    # Define a mapping of dataset names to their paths
    dataset_paths = {
        "youcook2": config['datasets']['youcook2'],
        "vatex": config['datasets']['vatex'],
        "tvc": config['datasets']['tvc'],
        "msrvtt": config['datasets']['msrvtt'],
        "msvd": config['datasets']['msvd'],
    }

    # Check if the dataset name is supported
    if dataset_name not in dataset_paths:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # flamingo use val set for youcook2
    if dataset_name == "youcook2":
        split = "val"
    # Set the paths based on the dataset name
    dataset_info = dataset_paths[dataset_name]
    video_dir_path = os.path.join(config['general']['data_root'], dataset_info['video_dir_path'])
    train_annotations_path = os.path.join(config['general']['data_root'], dataset_info['train_annotations_json_path'])
    val_annotations_path = os.path.join(config['general']['data_root'], dataset_info['val_annotations_json_path'])
    test_annotations_path = os.path.join(config['general']['data_root'], dataset_info['test_annotations_json_path'])


    train_dataset = VideoCaptionDataset(
        video_dir_path,
        train_annotations_path,
        split="train",
        dataset_name=dataset_name
    ) 

    print(f"Number of training samples: {len(train_dataset)}")

    test_dataset = VideoCaptionDataset(
        video_dir_path,
        test_annotations_path if split == "test" else val_annotations_path,
        split=split,
        dataset_name=dataset_name
    )

    if test_dataset is not None:
        print(f"Number of test samples: {len(test_dataset)}")

    effective_num_shots = num_shots if num_shots > 0 else 2
    # effective_num_shots = num_shots

    test_dataset = prepare_eval_samples(
        test_dataset,
        num_samples if num_samples > 0 else len(test_dataset),
        seed,
    )
    if num_samples > 0:
        print(f"Num shots, Num Inference Samples: {effective_num_shots}, {num_samples}")
    else:
        print(f"Num shots, Num Inference Samples: {effective_num_shots}, {len(test_dataset)}")

    query_set_size = min(query_set_size, len(train_dataset))

    in_context_samples = get_query_set(train_dataset, query_set_size, seed)

    predictions = defaultdict()

    for batch in more_itertools.chunked(
        tqdm(test_dataset, desc=f"Running Captioning inference {dataset_name.upper()} shots={num_shots}"),
        batch_size,
    ):
        batch_demo_samples = sample_batch_demos_from_query_set(
            in_context_samples, effective_num_shots, len(batch)
        )

        # print("Begin to generate context_exmples.......")
        batch_videos = []
        batch_text = []
        for i in range(len(batch)):
            if num_shots > 0:
                context_videos = [x["video"] for x in batch_demo_samples[i]]
            else:
                context_videos = []
            batch_videos.append(context_videos + [batch[i]["video"]])

            context_text = '<s>' + "".join(
                [
                    eval_model.caption_prompt(caption=x["caption"].strip())
                    for x in batch_demo_samples[i]
                ]
            )

            # Keep the text but remove the image tags for the zero-shot case, follow page 26 in flamingo
            if num_shots == 0:
                context_text = context_text.replace("<visual>", "")

            batch_text.append(context_text + eval_model.caption_prompt())
        #print(batch_text[0])
        # print("Begin to generate outputs.......")
            outputs = eval_model.get_video_outputs(
                batch_videos=batch_videos,
                batch_text=batch_text,
                max_generation_length=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )

        new_predictions = [
            postprocess_captioning_generation(out).replace('"', "") for out in outputs
        ]
        # print(new_predictions[0])
        for i, sample in enumerate(batch):
            predictions[sample["video_id"]] = {
                "caption": new_predictions[i],
            }

    # save the predictions to a temporary file
    results_path = f"{dataset_name}_results_{uuid.uuid4()}.json"

    with open(results_path, "w") as f:
        f.write(
            json.dumps(
                [
                    {"image_id": k, "caption": predictions[k]["caption"]} # 'image_id' is coco style
                    for k in predictions
                ],
                indent=4,
            )
        )

    metrics = compute_cider(
        result_path=results_path,
        annotations_path=test_annotations_path if split == "test" else val_annotations_path,
    )
    # delete the temporary file
    os.remove(results_path)
    return metrics
    # return metrics["CIDEr"] * 100.0
