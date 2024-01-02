HF Trainer do not support zero stage 3, 
look last part in "https://huggingface.co/docs/transformers/main/main_classes/deepspeed" for details.







Current loss minimum:
 https://github.com/facebookresearch/fairseq/issues/1529

 this two lines are for 

original:
"fp16_scale_tolerance": 0.25


"fp16_scale_tolerance": 0.25,
"min_loss_scale": 0.5

v2:

"fp16_scale_tolerance": 0.5,
"min_loss_scale": 0.2