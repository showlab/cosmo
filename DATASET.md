# Data Prparation

| Name | Category | Data |
|---|---|---|
|Pre-training|Image-Text|CC3M,LAION400M,DATACOMP1B|
||Video-Text|WebVid2.5M|
||Interleaved Image-Text|MMC4|
||Interleaved Video-Text|Howto-Internlink7M|
|Few-shot Evaluation|Image Captioning|COCO,FLICKR30K|
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
Please follow [TL;DR](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Too_Large_Data_Reduction_for_Vision-Language_Pre-Training_ICCV_2023_paper.pdf) to cluster and select samples.

## Interleaved Image-text Data
### MMC4


```bash
python src/utils/count_webdataset_sample.py --data_dir /datadrive_d/jinpeng/Code/videogpt4/datas/raw/mmc4/subdir
```



## Video-text Data
For the video-text data, we mainly use webvid dataset.
Please download raw video from [here](https://github.com/m-bain/webvid).
Then transform the data into webdataset format via:

```
data_preprocess/convert_webvid_to_wds.py
```


## Interleaved Video-text Data

Download Interleaved Video-text Dataset from [huggingface dataset](https://huggingface.co/datasets/Awiny/Howto-Interlink7M).

## Downstream Tasks



# Pre-training Data for COS-MOE(56B)


| Name | Category | Data |
|---|---|---|
|Pre-training|Image-Text|subset of {DATACOMP1B,LAION2B}|
||Interleaved Image-Text|MMC4,OBELICS|
