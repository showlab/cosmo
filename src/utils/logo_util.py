import os
import shutil
import json
import datetime
import multiprocessing
import platform
import GPUtil
import psutil
import re
from PIL import Image, ImageDraw, ImageFont
from termcolor import colored
import uuid
import torch.distributed as dist

class CustomLogger:
    def __init__(self, local_rank_for_log):
        self.local_rank_for_log = local_rank_for_log

    def info(self, message, color=None):
        if self.local_rank_for_log == 0:
            if color:
                print(colored(message, color))
            else:
                print(message)

    def info_w_delimiter(self, message, color=None):
        if self.local_rank_for_log == 0:
            self.info("*"*100, color=color)
            self.info(message, color=color)
            self.info("*"*100, color=color)

    def save_text_to_file(self, text, file_path="experiments/debug/case.txt"):
        if self.local_rank_for_log == 0:
            with open(file_path, "a") as f:
                f.write(text)
    
    def save_image_to_file(self, image, file_path="experiments/debug/case.png"):
        if self.local_rank_for_log == 0:
            image.save(file_path)

    def create_combined_image(self, text, images,  files_path="experiments/debug"):
        if self.local_rank_for_log == 0:
            # Convert the tensor images back to PIL images
            text_parts = text.split("<visual>")
            images_pil = images
            # Calculate the total width and height for the combined image
            total_width = sum(image.width for image in images_pil) + 20 * len(images_pil)
            max_height = max(image.height for image in images_pil)
            # Create a blank canvas for the combined image
            # Create a blank canvas for the combined image
            combined_image = Image.new('RGB', (total_width, max_height + 250), color='white')
            draw = ImageDraw.Draw(combined_image)
            # Add the text to the combined image, each part on a separate line
            font = ImageFont.load_default()
            text_position = (10, 10)  # Position of the text
            for i, text_part in enumerate(text_parts):
                draw.text(text_position, text_part, fill='black', font=font)
                text_width, text_height = draw.textsize(text_part, font=font)
                text_position = (10, text_position[1] + text_height + 10)  # Move to the next line

            # Add the images side by side to the combined image
            x_offset = 10
            for image in images_pil:
                combined_image.paste(image, (x_offset, 250))
                x_offset += image.width + 20
            image_filename = os.path.join(files_path, f"combined_image_{uuid.uuid4().hex}.jpg")
            combined_image.save(image_filename)

            return image_filename

def format_num(num):
    if num >= 1e9:
        return f'{num / 1e9:.2f}G'
    elif num >= 1e6:
        return f'{num / 1e6:.2f}M'
    elif num >= 1e3:
        return f'{num / 1e3:.2f}K'
    else:
        return str(num)

def extract_dataset_name(path):
    # Extract the dataset name using a regular expression
    match = re.search(r'/dataset/([^/]*)', path)
    if match:
        dataset_name = match.group(1)
        path_parts = path.split('/')
        # Check if there is a second-to-last segment, and if so, add it to the dataset name
        if len(path_parts) >= 3:
            second_last_segment = path_parts[-2]
            return dataset_name + '_' + second_last_segment
    return None

def device_info():
    # 1. cpu info
    # To print the number of CPUs
    print("Number of CPUs: ", multiprocessing.cpu_count())
    # To print CPU information
    print("CPU info: ", platform.processor())
    # 2. gpu info
    # Get the first available GPU
    gpus = GPUtil.getGPUs()
    gpu = gpus[0]

    print("Num of GPU: ", len(gpus))
    print("GPU RAM Total: ", gpu.memoryTotal)
    print("GPU RAM Free: ", gpu.memoryFree)
    print("GPU RAM Used: ", gpu.memoryUsed)
    print("GPU Utilization: ", gpu.load)
    # 3. memory info
    # Get system memory information
    mem_info = psutil.virtual_memory()

    print("Total Memory: ", mem_info.total)
    print("Available Memory: ", mem_info.available)
    print("Used Memory: ", mem_info.used)
    print("Memory Percentage used: ", mem_info.percent)

    return True

def save_config_and_src(config, src_dir, log_dir, suffix1="", suffix2="", add_time_stamp=True):
    """
    create a new dir in log_dir, and save config and src into it
    """
    lang_model_name = config["model_params"]["lang_model"]["name"]
    vision_encoder_name = config["model_params"]["vision_encoder"]["vision_encoder_name"]
    gpu_nums = str(dist.get_world_size()) + "gpus"
    elements = [lang_model_name, vision_encoder_name, gpu_nums]
    if suffix1:
        elements.append(suffix1.split("/")[-1].split(".")[0])
    if suffix2:
        elements.append(suffix2.split("/")[-1].split(".")[0])
    exp_string = '_'.join(elements)
    root_log_dir = os.path.join(log_dir, exp_string)
    os.makedirs(root_log_dir, exist_ok=True)
    if add_time_stamp:
        specific_log_dir = os.path.join(root_log_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
       specific_log_dir = root_log_dir
    # since only use time is hard to read, we split directory with dataset name
    os.makedirs(specific_log_dir, exist_ok=True)
    # Save config
    try:
        if not os.path.exists(os.path.join(specific_log_dir, "config.json")):
            with open(os.path.join(specific_log_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Unable to save config to {specific_log_dir}. Error: {e}")
        
    # Save src
    os.makedirs(os.path.join(specific_log_dir, "src"), exist_ok=True)
    for file in os.listdir(src_dir):
        source = os.path.join(src_dir, file)
        destination = os.path.join(specific_log_dir, "src", file)
        try:
            if not os.path.exists(destination):
                if os.path.isdir(source):
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)
            else:
                continue
        except Exception as e:
            print(f"Unable to copy file {source} to {destination}. Error: {e}")
            continue
    return specific_log_dir


def has_checkpoints(ckpt_path):
    """
    remain questions: sometimes rng_state.pth is not in the checkpoint dir while write not finished
    """
    if not os.path.exists(ckpt_path):
        return False

    checkpoints = []

    required_files = ["pytorch_model.bin", "training_args.bin", "trainer_state.json"]

    for dirname in os.listdir(ckpt_path):
        if dirname.startswith("checkpoint-"):
            checkpoint_num = int(dirname.split('-')[-1])  # Extract the number from "checkpoint-xxx"
            checkpoint_dir = os.path.join(ckpt_path, dirname)
            if os.path.isdir(checkpoint_dir) and all(os.path.exists(os.path.join(checkpoint_dir, fname)) for fname in required_files):
                checkpoints.append((checkpoint_num, checkpoint_dir))
    
    if checkpoints:
        checkpoints.sort(key=lambda x: x[0])  # Sort by the checkpoint number
        latest_checkpoint_dir = checkpoints[-1][1]  # Get the directory of the latest checkpoint
        print(f"Found latest checkpoint in {latest_checkpoint_dir}")
        return latest_checkpoint_dir
                
    return False