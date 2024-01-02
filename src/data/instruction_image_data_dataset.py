from .base_dataset import BaseDataset
from utils import *
import pandas as pd
import random
import os
import json
from PIL import Image
from subprocess import Popen, PIPE
from .base_dataset import BaseDataset
from .utils import split_json_data_by_node
from utils import *


class InstructionImageDataset(BaseDataset):
    def __init__(self, split, data_path, batch_size, tokenizer, image_processor, video_processor, dataset_params, custom_logger):
        super().__init__(split, data_path, batch_size, tokenizer, image_processor, video_processor, dataset_params, custom_logger)
        self.node_metadata = None
        self._load_metadata(data_path.strip())
        self.custom_logger.info(f"{self.split} Instruct Image Text Dataset num_samples: {len(self.metadata)}")
        self.custom_logger.info("batch size for image text dataset is: {}".format(self.batch_size))
        
    def _load_metadata(self, data_path):
        """
        The data_path may include a lot of json files, we need to read all of them
        """
        self.custom_logger.info(f"Loading metadata from {data_path}")
        json_files = data_path.split(';')  # Split the path by ';' to get individual JSON files
        self.metadata = pd.DataFrame()

        for json_file in json_files:
            with open(json_file, 'r') as file:
                json_data = json.load(file)
                # Assuming JSON structure contains 'id', 'image', and 'conversations'
                df = pd.DataFrame(json_data)  # Create a DataFrame from JSON data
                self.metadata = pd.concat([self.metadata, df], ignore_index=True)  # Concatenate using pd.concat

        if self.split_data_by_node:
            self.node_metadata = split_json_data_by_node(self.metadata)
        else:
            self.node_metadata = self.metadata
        if self.dataset_params['img_instruct']['MAX_SAMPLES'] == -1:
            print("Using all samples for img_instruct data")
        else:
            self.node_metadata = self.node_metadata[:self.dataset_params['img_instruct']['MAX_SAMPLES']]        

    def __len__(self):
        return len(self.node_metadata)

    def __getitem__(self, index):
        return self.get_suite(index)
    
    def _get_all_image_path(self):
        image_paths = [self._get_image_path(self.node_metadata.iloc[i])[0] for i in range(len(self.node_metadata))]
        return image_paths

    def _get_image_path(self, sample):
        rel_fp = sample['image']
        abs_fp = os.path.join(self.img_instruct_data_root, rel_fp)
        return abs_fp, rel_fp

    def get_raw_image(self, sample):
        abs_fp, rel_fp = self._get_image_path(sample)
        image = Image.open(abs_fp)
        return image
        
    def get_text(self, sample):
        """
        imagepath, caption or
        imagepath, caption,	gen_caption
        """
        data = sample['conversations']
        conversation = ""
        for item in data:
            if item['from'] == 'human':
                conversation += f"<human> {item['value']}\n"
            elif item['from'] == 'gpt':
                conversation += f"<gpt> {item['value']}\n"
        return conversation


    def get_image(self, sample):
        image = self.get_raw_image(sample)
        processed_image_tensor = self.preprocess_instruction_image_fn([image])
        return processed_image_tensor.squeeze(0)

    def get_suite(self, index):
        sample = self.node_metadata.iloc[index]
        image_tensor = self.get_image(sample)
        text = self.get_text(sample)
        ret = self.preprocess_instruction(image_tensor, text)
        # while result is None:
        #     sample = self.node_metadata.iloc[index]
        #     # print(sample)
        #     try:
        #         image_tensor = self.get_image(sample)
        #         text = self.get_text(sample)
        #         ret = self.preprocess_instruction(image_tensor, text)
        #         result = True
        #     except Exception as e:
        #         self.custom_logger.info(f"Error while read file idx {sample[0]} -> {e}")
        #         index = random.randint(0, len(self.node_metadata) - 1)
        return ret