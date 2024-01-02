# Data Prparation

| Name | Category | Data |
|---|---|---|
|Pre-training|Image-Text|CC3M,LAION400M,DATACOMP1B|
||Video-Text|WebVid2.5M|
||Interleaved Image-Text|CC3M|
||Interleaved Video-Text|Howto-Interlin7M|
|Few-shot Evluation|Image Captioning|COCO,FLICKR30K|
||Image QA|OK-VQA,VIZWIZ,VQA-V2,TextVQA|
||Image Classification|HatefulMems,DataComp|
||Image Retrieval|DataComp|
||Video Captioning|TVC,MSVD,MSRVTT,YouCook2,VATEX|
||Video VQA|MSRVTT,MSVD,TGIF|
|Instruction Tuning||ShareGpt4V|


All the pre-training data is in the format tar file. For example

```
MMC4/000000.tar,
...
MMC4/001244.tar
```
## Image-text Data

## Interleaved Image-text Data
### MMC4

```bash
python src/utils/count_webdataset_sample.py --data_dir /datadrive_d/jinpeng/Code/videogpt4/datas/raw/mmc4/subdir
```
