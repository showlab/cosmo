# Instruction Tuning

## Data Preparation
The only difference between instruction data and normal image-text pairs is design the input.


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

## Fine-tuning

```
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun src/main_instruction_tuning.py --base_config "src/config/tuning/base.yaml" \
--deepspeed "src/config/deepspeed/deepspeed_config_mistral.json"
```