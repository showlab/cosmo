from eval.models import cosmo
import yaml
import torch
import argparse
from PIL import Image
import cv2
import numpy as np
from utils.logo_util import CustomLogger
from src.eval.eval_tasks.utils.coco_metric import postprocess_captioning_generation

model_mapping = {
    "cosmo": cosmo.EvalModel,
    # add more models as needed
}


def predict(eval_model, image_path, text, max_generation_length, num_beams, length_penalty):
    print(f"Predicting for image {image_path} with prompt {text}")
    image = Image.open(image_path)
    image.load()
    
    batch_images = []
    batch_text = []
    batch_images.append([image])
    print(eval_model.vqa_prompt(text))
    batch_text.append(eval_model.vqa_prompt(text))
    prompt = "<visual> Describe this image:"
    batch_text.append(prompt)

    outputs = eval_model.get_outputs(
        batch_images=batch_images,
        batch_text=batch_text,
        max_generation_length=max_generation_length,
        num_beams=num_beams,
        length_penalty=length_penalty,
    )

    new_predictions = [
        postprocess_captioning_generation(out).replace('"', "") for out in outputs
    ]

    print(new_predictions)

    # the batch_images generally include one image
    # show the image and the generated caption in the same window and save with opencv
    # then save the image with the generated caption into directory "output_images"
    for i in range(len(batch_images[0])):
        img = batch_images[0][i]
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.putText(
            img,
            batch_text[i]+new_predictions[i],
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(f"output_images/{image_path.split('/')[-1]}", img)


    return new_predictions

def main(args):
    custom_logger = CustomLogger(args.local_rank)
    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    eval_model = model_mapping["cosmo"](config, custom_logger, device)
    result = predict(eval_model, args.image_path, args.prompt, 10, 5, -0.2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",  default='src/config/generation/generation_opt_config_local.yaml', help="Path to evaluate config.yaml")
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--image_path", type=str, default="test1.jpg")
    parser.add_argument("--video_path", type=str, default="yest1.mp4")
    parser.add_argument("--prompt", type=str, default="This is a photo of")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)