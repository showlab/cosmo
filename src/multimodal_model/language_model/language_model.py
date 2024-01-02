import random
import yaml
import importlib
import torch.nn as nn
from .base_model.knn_memory import KNNMemory
from .base_model.knn_attention import KNNAttention

with open('src/config/model_version/model_version.yaml') as f:
    config = yaml.safe_load(f)
model_version = config['language_model_helper']['version']
model_module = importlib.import_module(f'multimodal_model.language_model.model_helper{model_version}')
GatedCrossAttentionBlock = getattr(model_module, 'GatedCrossAttentionBlock')
getattr_recursive = getattr(model_module, 'getattr_recursive')
setattr_recursive = getattr(model_module, 'setattr_recursive')



class MultiModalityDecoderLayer(nn.Module):
    def __init__(self, gated_cross_attn_layer, decoder_layer, memory_layer=None, knn_memory=None, gradient_checkpointing=False):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.memory_layer = memory_layer
        self.knn_memory = knn_memory
        self.vis_x = None
        self.media_locations = None
        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer._use_gradient_checkpointing = (
                gradient_checkpointing
            )
        self.decoder_layer._use_gradient_checkpointing = gradient_checkpointing

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, vis_x):
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations):
        self.media_locations = media_locations

    def condition_use_cached_media(self, use_cached_media):
        self.use_cached_media = use_cached_media
    
    def forward(
        self,
        lang_x,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        # print("lang_x.shape", lang_x.shape)
        if self.gated_cross_attn_layer is None:
            return self.decoder_layer(
                lang_x, attention_mask=attention_mask, **decoder_layer_kwargs
            )

        if self.vis_x is None:
            raise ValueError("vis_x must be conditioned before forward pass")

        if self.media_locations is None:
            raise ValueError("media_locations must be conditioned before forward pass")
        if self.memory_layer is not None and self.knn_memory is not None:
            lang_x = self.memory_layer(lang_x, knn_memory=self.knn_memory)
        
        lang_x = self.gated_cross_attn_layer(
            lang_x,
            self.vis_x,
            media_locations=self.media_locations,
            use_cached_media=self.use_cached_media,
        )
        lang_x = self.decoder_layer(
            lang_x, attention_mask=attention_mask, **decoder_layer_kwargs
        )
        return lang_x


class MultiModalityLM(nn.Module):
    """
    uni modal text decoder + multimodal decoder
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_add_multimodality_attention(
        self,
        media_token_id,
        lang_hidden_size,
        vis_hidden_size,
        uni_modal_layers,
        interval_layer,
        cross_attention_compress_ratio,
        gradient_checkpointing,
        use_memory_layer,
        only_attend_immediate_media,
        qv_norm,
    ):
        """
        Add multimodal attention to the decoder layers. Also the last layer.
        Store the media token id for computing the media locations.
        """
        self.old_decoder_blocks = self._get_decoder_layers()
        self.gated_cross_attn_layers = nn.ModuleList(
            [
                GatedCrossAttentionBlock(
                    dim=lang_hidden_size, dim_visual=vis_hidden_size, compress_ratio=cross_attention_compress_ratio,
                    only_attend_immediate_media=only_attend_immediate_media,
                    qv_norm=qv_norm,
                )
                if (layer_idx >= uni_modal_layers and (layer_idx-uni_modal_layers) % interval_layer == 0)
                # if (layer_idx >= uni_modal_layers and (layer_idx-uni_modal_layers) % interval_layer == 0) or (layer_idx==len(self._get_decoder_layers())-1) or (layer_idx==uni_modal_layers)
                else None
                for layer_idx, _ in enumerate(self._get_decoder_layers())
            ]
        )
        print("{} layers in total, include {} unimodal layers and {} cross_attention layers".format(len(self._get_decoder_layers()), uni_modal_layers, sum(1 for item in self.gated_cross_attn_layers if item is not None)))
        if use_memory_layer:
            # Get the first gated_cross_attn_layer that's not None
            first_gated_cross_attn_layer = next((layer for layer in self.gated_cross_attn_layers if layer is not None), None)
            
            # Ensure that you've found a non-None layer before accessing its attributes
            if first_gated_cross_attn_layer:
                self.memory_layer = KNNAttention(
                    dim=first_gated_cross_attn_layer.dim
                )
                self.knn_memory = KNNMemory(
                    dim=first_gated_cross_attn_layer.dim_head,
                    max_memories=64000,
                )
            else:
                raise ValueError("No valid GatedCrossAttentionBlock found!")
        else:
            self.memory_layer = None
            self.knn_memory = None
        # Find out the layer_idx after the first appearance of GatedCrossAttentionBlock
        first_cross_attn_idx = next((idx for idx, layer in enumerate(self.gated_cross_attn_layers) if layer is not None), None)
        if use_memory_layer:
            print("add a memory layer before the {}-th layer".format(first_cross_attn_idx+1))
            new_decoder_layers = []
            for layer_idx, (gated_cross_attn_layer, decoder_layer) in enumerate(zip(self.gated_cross_attn_layers, self.old_decoder_blocks)):
                if layer_idx == first_cross_attn_idx:
                    # Add memory layer only for this specific layer
                    new_decoder_layers.append(
                        MultiModalityDecoderLayer(
                            gated_cross_attn_layer,
                            decoder_layer,
                            self.memory_layer,
                            self.knn_memory,
                            gradient_checkpointing=gradient_checkpointing,
                        )
                    )
                else:
                    new_decoder_layers.append(
                        MultiModalityDecoderLayer(
                            gated_cross_attn_layer,
                            decoder_layer,
                            gradient_checkpointing=gradient_checkpointing,
                        )
                    )
            self._set_decoder_layers(nn.ModuleList(new_decoder_layers))
        else:
            self._set_decoder_layers(
                nn.ModuleList(
                    [
                        MultiModalityDecoderLayer(
                            gated_cross_attn_layer, decoder_layer, gradient_checkpointing=gradient_checkpointing,
                        )
                        for gated_cross_attn_layer, decoder_layer in zip(self.gated_cross_attn_layers, self.old_decoder_blocks)
                    ]
                )
            )
        self.media_token_id = media_token_id
        self.initialized_multimodality_attention = True
        self._use_cached_vision_x = False

    def forward(self, input_ids, attention_mask, **kwargs):
        """Condition the videogpt4 layers on the media locations before forward()"""
        if not self.initialized_multimodality_attention:
            raise ValueError(
                "Multi Modality Decoder layers are not initialized. Please call `init_add_multimodality_attention` first."
            )

        media_locations = input_ids == self.media_token_id

        # from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/flamingo_lm.py
        # if there are media already cached and we're generating and there are no media tokens in the input,
        # we'll assume that ALL input tokens should attend to the last previous media that is cached.
        # this is especially important for HF generate() compatibility, since generate() calls forward()
        # repeatedly one token at a time (with no media tokens).
        # without this check, the model would not attend to any images when generating (after the first token)
        use_cached_media_locations = (
            self._use_cached_vision_x
            and self.is_conditioned()
            and not media_locations.any()
        )

        for layer in self._get_decoder_layers():
            if not use_cached_media_locations:
                layer.condition_media_locations(media_locations)
            layer.condition_use_cached_media(use_cached_media_locations)

        # Convert attention_mask to Boolean dtype for MPT model
        attention_mask = attention_mask.bool()

        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        return super().forward(**kwargs)  # Call the other parent's forward method

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)
            layer.condition_use_cached_media(None)
