# Instruction Tuning

## Data Preparation
The only difference between instruction data and normal image-text pairs is design the input.
Follow [IDELICS](https://huggingface.co/HuggingFaceM4/idefics-80b-instruct), we fine-tuning the model with M3IT, LRV-Instruction, LLaVA-Instruct, SVIT. 


## Format
Prepare json file with elements looks like:

```
    {
        "id": "000000033471",
        "image": "coco/train2017/000000033471.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat are the colors of the bus in the image?"
            },
            {
                "from": "gpt",
                "value": "The bus in the image is white and red."
            },
            {
                "from": "human",
                "value": "What feature can be seen on the back of the bus?"
            },
        ]
    },
```

The example of script to process dataset is list in data_preprocess/m3it_preprocess.py.




## Fine-tuning

```
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun src/main_instruction_tuning.py --base_config "src/config/tuning/base.yaml" \
--deepspeed "src/config/deepspeed/deepspeed_config_mistral.json"
```