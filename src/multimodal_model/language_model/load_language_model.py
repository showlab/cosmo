# part of implementation comes from open-flamingo (https://github.com/mlfoundations/open_flamingo)
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import importlib
import torch
# from .language_model import MultiModalityLM

with open('src/config/model_version/model_version.yaml') as f:
    config = yaml.safe_load(f)
model_version = config['language_model_helper']['version']
model_module = importlib.import_module(f'multimodal_model.language_model.model_helper{model_version}')
extend_instance = getattr(model_module, 'extend_instance')

language_model_version = config['language_model']['version']
language_model_module = importlib.import_module(f'multimodal_model.language_model.language_model{language_model_version}')
MultiModalityLM = getattr(language_model_module, 'MultiModalityLM')

__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
    "mistral": "model.layers",
    "mixtral": "model.layers",
    "persimmon": "model.layers",
}

def _infer_decoder_layers_attr_name(model):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


def load_language_model(
    vis_features_dim: int,
    lang_model_path: str,
    tokenizer_path: str,
    uni_modal_layers: int = 12,
    interval_layer: int = 1,
    cross_attention_compress_ratio: int = 1,
    decoder_layers_attr_name: str = None,
    use_local_files: bool = True, # False for default
    gradient_checkpointing: bool = False,
    use_memory_layer: bool = False,
    only_attend_immediate_media: bool = True,
    qv_norm: bool = False,
    **videogpt4_kwargs,
):
    """
    Initialize a multi-modality language model.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        lang_model_path (str): path to pretrained language encoder, HuggingFace model hub name, or Flax checkpoint
        tokenizer_path (str): path to pretrained tokenizer, HuggingFace model hub name, or Flax checkpoint
        uni_modal_layers (int, optional): how many layers for text decoder only. Defaults to 12.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Tokenizer: A tokenizer for the language model
    """
    print(f"Loading language model from {lang_model_path}")
    print(f"Loading tokenizer from {tokenizer_path}")
    if 'llama' in lang_model_path:
        text_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=use_local_files, trust_remote_code=True, use_fast=False,
            unk_token="<unk>", bos_token="<s>", eos_token="</s>"
        )
    else:
        text_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=use_local_files, trust_remote_code=True,
        )
    # add Flamingo special tokens to the tokenizer, for mmc4 training
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<visual>", "<|beginofchunk|>", "<human>", "<gpt>"]}
    )
    # if text_tokenizer.pad_token is None:
    # Issue: GPT models don't have a pad token, which we use to
    # modify labels for the loss.
    text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    # Set the tokenizer's pad_token attribute to "<PAD>"
    text_tokenizer.pad_token = "<PAD>"
    # mixtral 7x8b is too large to fit in memory, read with 4 bit
    if 'mixtral' in lang_model_path.lower():
        import os
        print("loading mixtral with 4 bit")
        from transformers import AutoModel, AutoConfig
        multimodal_decoder = AutoModelForCausalLM.from_pretrained(
            lang_model_path, local_files_only=use_local_files, output_hidden_states=True, trust_remote_code=True,
            # torch_dtype=torch.float16
            load_in_4bit=True
            ).cuda()
    else:
        multimodal_decoder = AutoModelForCausalLM.from_pretrained(
            lang_model_path, local_files_only=use_local_files, output_hidden_states=True, trust_remote_code=True,
        )

    # print layer names
    # for name, param in multimodal_decoder.named_parameters():
    #     print(name)

    # hacks for MPT-1B, which doesn't have a get_input_embeddings method
    if "mpt-1b-redpajama-200b" in lang_model_path:

        class EmbeddingFnMixin:
            def get_input_embeddings(self):
                return self.transformer.wte

            def set_input_embeddings(self, new_embeddings):
                self.transformer.wte = new_embeddings

        extend_instance(multimodal_decoder, EmbeddingFnMixin)

    if hasattr(multimodal_decoder.config, "d_model"):
        lang_features_dim = multimodal_decoder.config.d_model  # mpt uses d_model
    else:
        lang_features_dim = multimodal_decoder.config.hidden_size
    extend_instance(multimodal_decoder, MultiModalityLM)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name( multimodal_decoder)
    multimodal_decoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    multimodal_decoder.resize_token_embeddings(len(text_tokenizer))

    # text_tokenizer.encode("<|endofchunk|>")[-1],
    # media_token_id = text_tokenizer.encode("<visual>")[-1],
    media_token_id = text_tokenizer("<visual>", add_special_tokens=False)["input_ids"][-1]
    # initalize multimodal cross-attention layers
    multimodal_decoder.init_add_multimodality_attention(
        media_token_id=media_token_id,
        lang_hidden_size=lang_features_dim,
        vis_hidden_size=vis_features_dim,
        uni_modal_layers=uni_modal_layers,
        interval_layer=interval_layer,
        cross_attention_compress_ratio=cross_attention_compress_ratio,
        gradient_checkpointing=gradient_checkpointing,
        use_memory_layer=use_memory_layer,
        only_attend_immediate_media=only_attend_immediate_media,
        qv_norm=qv_norm,
    )
    return multimodal_decoder, text_tokenizer, lang_features_dim
