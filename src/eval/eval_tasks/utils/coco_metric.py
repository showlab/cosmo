from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO


def compute_cider(
    result_path,
    annotations_path,
):
    # create coco object and coco_result object
    coco = COCO(annotations_path)
    coco_result = coco.loadRes(result_path)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params["image_id"] = coco_result.getImgIds()
    coco_eval.evaluate()

    return coco_eval.eval

def postprocess_captioning_generation(predictions):
    return predictions.split("Output", 1)[0]


# unit test
if __name__ == "__main__":
    # test
    result_json_path = "/datadrive_d/jinpeng/Code/videogpt4/vatex_results_1d3b5445-235b-4f83-993e-f295bc36d0ed.json"
    annotation_json_path = "/home/jinpeng/blob/vigstandard_data/v-jinpewang/dataset/downstream_datasets/vatex/annotations/validation_w_id_coco_style_data.json"

    acc = compute_cider(result_json_path, annotation_json_path)
    print(acc)