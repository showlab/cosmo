## 2. Evaluation of Few-shot Learning on Downstream Tasks Without Additional Training


- Image-text VQA

- Image-text Captioning

- Image-text Retrieval

- Image Classification

- Video-text VQA

- Video-text Captioning

- Video-text MC

- 38 image-text datasets over datacomp


Simply, can test all tasks with one run

```python
python src/main_eval_multiprocess.py  --base_config src/config/eval/eval_base_multiprocess.yaml --variant_config src/config/eval/mistral/7b_base_local.yaml --world_size 8
```
