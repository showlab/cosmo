# become the m3it dataset into json file
from datasets import load_dataset, load_from_disk
import os
import json
from tqdm import tqdm
import base64
# Define paths
# dataset_path = "/home/jinpeng/blob/vigstandard_data/v-jinpewang/dataset/instruction_tuning_data/M3IT"
dataset_path = "MMInstruction/M3IT"
save_path = "instruction_tuning_data/M3IT/organized_data"
os.makedirs(save_path, exist_ok=True)


dataset_names = [
    "coco",
    "textcap",
    "image-paragraph-captioning",
    "coco-goi",
    "coco-text",
    "imagenet",
    "coco-itm",
    "snli-ve",
    "mocheg",
    "iqa",
    "vqa-v2",
    "shapes",
    "docvqa",
    "ocr-vqa",
    "st-vqa",
    "text-vqa",
    "gqa",
    "okvqa",
    "a-okvqa",
    "science-qa",
    "viquae",
    "clevr",
    "nlvr",
    "vcr",
    "visual-mrc",
    "winoground",
    "vist",
    "visual-dialog",
    "multi30k",
    "fm-iqa",
    "coco-cn",
    "flickr8k-cn",
    "chinese-food",
    "mmchat",
    "ss",
    "ivqa",
    "msvd-qa",
    "activitynet-qa",
    "msrvtt",
    "msrvtt-qa"
]


output_list = []

# Assuming dataset_names, dataset_path, and save_path are defined elsewhere in your script
for dataset_name in dataset_names:
    print("Processing", dataset_name, "...")
    dataset = load_dataset(dataset_path, dataset_name)["train"]
    data = list(dataset)
    json_save_path = os.path.join(save_path, f"{dataset_name}_train.json")
    img_save_path = os.path.join(save_path, f"{dataset_name}_train")

    # Ensure the directory exists
    os.makedirs(img_save_path, exist_ok=True)
    
    output_list = []  # Initialize output_list for storing modified items

    # Transform dataset items
    for index, item in enumerate(tqdm(data)):
        # Extract and prepare data
        question = item["instruction"] + item["inputs"]
        answer = item["outputs"]
        base64_image = item["image_base64_str"][0]  # Do this before popping
        # Pop unused keys
        item.pop("instruction")
        item.pop("inputs")
        item.pop("outputs")
        item.pop("image_base64_str")
        item["id"] = index
        image_path = os.path.join(img_save_path, f"{item['id']}.jpg")
        rel_image_path = '/'.join(image_path.split('/')[-4:])
        
        # Create a new item structure
        new_item = {
            "id": item["id"],
            "image": rel_image_path,
            "image_base64_str": base64_image,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n" + question
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        output_list.append(new_item)

    # Save the transformed data to a JSON file
    with open(json_save_path, "w") as f:
        json.dump(output_list, f, ensure_ascii=False, indent=4)

    print(f"Processed {len(data)} items.")