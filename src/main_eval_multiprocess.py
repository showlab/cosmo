import argparse
import yaml
import torch
import numpy as np
import random
from collections import defaultdict
import json
import torch.multiprocessing as mp
from torch.multiprocessing import Manager

from eval import evaluate_model, cosmo
from utils import CustomLogger

model_mapping = {
    "cosmo": cosmo.EvalModel,
    # add more models as needed
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_config(base_path, variant_path):
    with open(base_path, 'r') as base_file:
        config = yaml.safe_load(base_file)
    if variant_path is None:
        return config
    with open(variant_path, 'r') as variant_file:
        variant_config = yaml.safe_load(variant_file)

    # Recursively update the base config with the variant config
    def update_config(base, variant):
        for key, value in variant.items():
            if isinstance(value, dict):
                base[key] = update_config(base.get(key, {}), value)
            else:
                base[key] = value
        return base

    return update_config(config, variant_config)

def convert_to_results_file(checkpoint_path):
    # Split the path into parts
    parts = checkpoint_path.split('/')
    
    # Extract the directory path up to the timestamp folder
    directory_path = '/'.join(parts[:-2])
    
    # Extract the checkpoint number from the "checkpoint-XXXXX" folder
    checkpoint_number = parts[-2].split('-')[-1]
    
    # Combine the directory path with the checkpoint number and "_eval_results.txt"
    result_file_path = f"{directory_path}/{checkpoint_number}_eval_results.txt"
    
    return result_file_path

def evaluate_single_task(config, eval_model, task_name, dataset_name, shot, seed, custom_logger, gpu, shared_results):
    set_seed(seed)  # set seed for reproducibility
    with torch.no_grad():
        accuracy = evaluate_model(config, eval_model, task_name, dataset_name, shot)
    print('*'*50)
    print(f'Accuracy for {task_name} on {dataset_name} dataset for shot {shot}: {accuracy}')
    print('*'*50)
    # save results somewhere
    result = {task_name: {dataset_name: {'shot': shot, 'seed': seed, 'accuracy': accuracy}}}
    shared_results.append(result) # Add result to shared list

def evaluate_tasks_single_gpu(world_size, rank, args, task_list, config, custom_logger, shared_results):
    device = torch.device("cuda", rank)  # Assign a specific GPU to each process based on its rank
    eval_model = model_mapping[config['model_name']](config, custom_logger, device)
    for task_name, dataset_name, shot, seed in task_list:
        print(f'!!!!!!!!Rank {rank} evaluating {task_name} on {dataset_name} dataset for shot {shot} with seed {seed}')
        evaluate_single_task(config, eval_model, task_name, dataset_name, shot, seed, custom_logger, rank, shared_results)
        # try:
        #     evaluate_single_task(config, eval_model, task_name, dataset_name, shot, seed, custom_logger, rank, shared_results)
        # except Exception as e:
        #     print(f'!!!!!!!!Rank {rank} failed to evaluate {task_name} on {dataset_name} dataset for shot {shot} with seed {seed}')
        #     print(f'!!!!!!!!Rank {rank} exception: {e}')
        #     continue

def process_shared_results(shared_results):
    processed_results = defaultdict(lambda: defaultdict(list))
    
    for result_item in shared_results:
        for task_name, task_data in result_item.items():
            for dataset_name, dataset_data in task_data.items():
                processed_results[task_name][dataset_name].append(dataset_data)

    return processed_results

def write_results(final_results, results_file):
    with open(results_file, "a") as f:
        # Iterate through tasks
        for task_name in sorted(final_results.keys()):
            f.write(f"Task: {task_name}\n")

            # Iterate through datasets for each task
            for dataset_name in sorted(final_results[task_name].keys()):
                f.write(f"Dataset: {dataset_name}\n")

                # Sort results by shot and write
                sorted_results_by_shot = sorted(final_results[task_name][dataset_name], key=lambda x: x['shot'])
                for result in sorted_results_by_shot:
                    json.dump(result, f)
                    f.write('\n') # Writes an empty line after the JSON object

def main(args):
    print("Begin to evaluate...")
    custom_logger = CustomLogger(0)
    config = load_config(args.base_config, args.variant_config)
    tasks_config = config.get("tasks")
    results_file = convert_to_results_file(config['general']['ckpt_path'])
    with open(results_file, "a") as f:
        f.write('Begin to write json files...\n')  # Writes an empty line after the JSON object
        json.dump(config, f)
        f.write('\n')  # Writes an empty line after the JSON object
    # Start each process for each GPU
    world_size = args.world_size
    task_list = []
    for task_info in tasks_config:
        task_name = task_info.get("name")
        datasets_for_task = task_info.get("datasets")
        for dataset_name in datasets_for_task:
            # for each task and dataset, inference num_trails X num_shots times
            for shot in config['general']['shots']:
                for seed, trial in zip(config['general']['trial_seeds'], range(config['general']['num_trials'])):
                    if task_name == "retrieval" and shot != 0:
                        continue
                    if task_name == "zs_classification" and shot != 0:
                        continue
                    task_list.append((task_name, dataset_name, shot, seed))

    num_tasks_per_gpu = (len(task_list) + world_size - 1) // world_size
    task_list_per_gpu = [task_list[i*num_tasks_per_gpu:(i+1)*num_tasks_per_gpu] for i in range(world_size)]

    # Initialize the multiprocessing context
    mp.set_start_method('spawn')

    manager = Manager()
    shared_results = manager.list() # Shared list across all processes


    processes = []
    for rank in range(world_size):
        p = mp.Process(target=evaluate_tasks_single_gpu, args=(world_size, rank, args, task_list_per_gpu[rank], config, custom_logger, shared_results))
        p.start()
        processes.append(p)


    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Now, process the shared results and write them in the desired order
    final_results = process_shared_results(shared_results)
    write_results(final_results, results_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config', default='src/config/eval/eval_base_multiprocess.yaml')
    parser.add_argument('--variant_config', type=str, default=None)
    parser.add_argument("--world_size", type=int, default=4, help="The number of GPUs you have.")
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--deepspeed', type=str, default='src/config/deepspeed/deepspeed_config.json')
    args = parser.parse_args()

    main(args)