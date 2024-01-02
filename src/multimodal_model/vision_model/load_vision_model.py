import open_clip
from transformers import SamModel, SamProcessor
import torch
from .base_model.mediasparseformer import SparseFormer

def load_vision_model(
    vision_encoder_name: str,
    vision_encoder_arch: str,
    vision_encoder_pretrained: str,
    ckpt_path: str,
    cache_dir: str,
    custom_augment: bool,
    **videogpt4_kwargs,
):
    """
    Initalize a vision encoder model.
    Args:
        vision_encoder_name (str): name of vision encoder
        vision_encoder_arch (str): path to pretrained clip model (e.g. "ViT-B-32"), only suitable for clip vision encoder
        vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k"), only suitable for clip vision encoder
        ckpt_path (str): path to pretrained model, only suitable for sparseformer vision encoder
    Returns:
        vision_encoder: vision encoder model
        image_processor: pipeline to preprocess input images
        vis_dim: dimension of visual features
    """
    # set the vision encoder to output the visual features
    if vision_encoder_name == "clip":
        print("Loading CLIP vision encoder")
        vision_encoder, train_image_processor, val_image_processor = open_clip.create_model_and_transforms(
            vision_encoder_arch, pretrained=vision_encoder_pretrained,
            cache_dir=cache_dir
        )
        vision_encoder.visual.output_tokens = True
        vis_dim=open_clip.get_model_config(vision_encoder_arch)["vision_cfg"][
            "width"]
    elif vision_encoder_name == "sam":
        print("Loading SAM vision encoder")
        # "facebook/sam-vit-base"
        vision_encoder = SamModel.from_pretrained(vision_encoder_arch)
        image_processor = SamProcessor.from_pretrained(vision_encoder_pretrained)
        vis_dim = 2048
    elif vision_encoder_name == "sparseformer":
        print("Loading SparseFormer vision encoder")
        clip_model, train_image_processor, val_image_processor = open_clip.create_model_and_transforms(
            vision_encoder_arch, pretrained=vision_encoder_pretrained)
        vision_encoder = SparseFormer(
            conv_dim=64,
            num_latent_tokens=64,
            token_sampling_points=16,
            width_configurations=[384, 1024],
            drop_path_rate=0.0,
            parent_vit_model=clip_model.visual)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        state_dict = state_dict['model'] if 'model' in state_dict else state_dict
        vision_encoder.load_2d_state_dict(state_dict)
        vis_dim = 1024
    else:
        raise ValueError(f"Unknown vision encoder name: {vision_encoder_name}")
    if custom_augment:
        return vision_encoder, None, vis_dim
    else:
        return vision_encoder, [train_image_processor, val_image_processor], vis_dim
