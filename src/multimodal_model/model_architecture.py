"""
gather tensors from all gpus for contrastive loss.
"""
import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
import torch.distributed as dist
import importlib
import yaml
from .model_helper import EmbedToLatents, LayerNorm
# from .base_model.knn_memory import KNNMemory
with open('src/config/model_version/model_version.yaml') as f:
    config = yaml.safe_load(f)
model_version = config['vision_model_helper']['version']
model_module = importlib.import_module(f'multimodal_model.vision_model.model_helper{model_version}')
PerceiverResampler = getattr(model_module, 'PerceiverResampler')
PerceiverAttention = getattr(model_module, 'PerceiverAttention')
FeedForward = getattr(model_module, 'FeedForward')



class CosMo(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_model: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        vis_dim: int,
        text_dim: int = 2048,
        uni_modal_layers: int = 12,
        dim_latents: int = 1024,
        contrastive_temperature: float = 1.0,
        vision_encoder_name: str = 'clip',
        contrastive_gather_way: str = 'single_gpu',
        use_text_memory: bool = False,
        qv_norm: bool = False,
    ):
        """
        Args:
            vision_encoder (nn.Module): CLIP/SAM
            lang_model (nn.Module):causal language model like OPT/LLAMA
            eoc_token_id (int): Token id for <|endofchunk|>
            media_token_id (int): Token id for <visual>
            vis_dim (int): Dimension of the visual features.
                Visual features are projected to match this shape along the last dimension.
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
            use_media_placement_augmentation (bool, optional): Whether to randomly assign images to the preceding or following text in training. Defaults to False.
        """
        super().__init__()
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.vis_dim = vis_dim
        self.text_dim = text_dim
        self.uni_modal_layers = uni_modal_layers
        self.dim_latents = dim_latents
        self.vision_encoder = vision_encoder
        if vision_encoder_name == 'sparseformer':
            self.perceiver = None
        else:
            self.perceiver = PerceiverResampler(dim=self.vis_dim)
        self.lang_model = lang_model
        # ===============The below for contrastivce loss====================
        self.img_to_latents = EmbedToLatents(self.vis_dim, self.dim_latents)
        self.text_to_latents = EmbedToLatents(self.text_dim, self.dim_latents)
        self.temperature = nn.Parameter(torch.Tensor([contrastive_temperature]))
        self.ce = F.cross_entropy
        self.text_cls_norm = LayerNorm(self.text_dim)
        self.text_learn_attnetion_layer = PerceiverAttention(dim=self.text_dim, qv_norm=qv_norm)
        self.text_latent = nn.Parameter(torch.randn(1, self.text_dim)) 
        self.text_learn_ff = FeedForward(dim=self.text_dim)
        self.groups = None
        self.vision_encoder_name = vision_encoder_name
        self.contrastive_gather_way = contrastive_gather_way
        # ==============The below for longer text generation=================
        # self.use_text_memory = use_text_memory
        # if self.use_text_memory:
        #     self.text_memory = KNNMemory(
        #         dim = 512,                  # dimension of key / values
        #         max_memories = 64000,       # maximum number of memories to keep (will throw out the oldest memories for now if it overfills)
        #         num_indices = 64            # this should be equivalent to batch dimension, as each batch keeps track of its own memories, expiring when it sees a new document
        #     )

    def init_group(self):
        world_size = dist.get_world_size()
        num_gpus_per_node = torch.cuda.device_count()
        groups = []
        for j in range(world_size//num_gpus_per_node):
            node_ranks = [j * num_gpus_per_node + i for i in range(num_gpus_per_node)]
            node_ranks = [rank for rank in node_ranks if rank < world_size]
            group = dist.new_group(ranks=node_ranks)
            groups.append(group)
        return groups


    def _compute_contrastive_loss(self, vision_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Computes the contrastive loss for the given vision embeddings and model output.
        Compute on single_gpu as default.
        """
        text_embeds = self.text_to_latents(text_embeds)
        vision_embeds = self.img_to_latents(vision_embeds)
        if self.contrastive_gather_way == 'all_nodes':
            text_latents = gather_tensors_all_nodes(text_embeds)
            image_latents = gather_tensors_all_nodes(vision_embeds)
        elif self.contrastive_gather_way == 'single_node' and dist.is_initialized():
            if self.groups is None:
                self.groups = self.init_group()
            group = self.groups[dist.get_rank()//torch.cuda.device_count()]
            text_latents = group_all_gather(text_embeds, group=group, group_size=torch.cuda.device_count(), dim=0)
            image_latents = group_all_gather(vision_embeds, group=group, group_size=torch.cuda.device_count(), dim=0)
        else:
            text_latents = text_embeds
            image_latents = vision_embeds
        text_latents = F.normalize(text_latents, dim=-1)
        image_latents = F.normalize(image_latents, dim=-1)

        # sim = einsum('i d, j d -> i j', text_latents, image_latents)
        sim = torch.matmul(text_latents, image_latents.t())
        sim = sim * self.temperature.exp()
        batch = image_latents.shape[0]
        contrastive_labels = torch.arange(batch, device=sim.device)
        contrastive_loss = (self.ce(sim, contrastive_labels) + self.ce(sim.t(), contrastive_labels)) * 0.5

        k = min(5, batch)
        # Compute the top-1 and top-5 accuracies for text-to-image matching
        top1_accuracy_txt2img = (sim.argmax(dim=1) == contrastive_labels).float().mean()
        topk_accuracy_txt2img = (sim.topk(k, dim=1).indices == contrastive_labels.view(-1, 1)).float().sum(dim=1).mean()

        # Compute the top-1 and top-5 accuracies for image-to-text matching
        sim_transpose = sim.t()
        top1_accuracy_img2txt = (sim_transpose.argmax(dim=1) == contrastive_labels).float().mean()
        topk_accuracy_img2txt = (sim_transpose.topk(k, dim=1).indices == contrastive_labels.view(-1, 1)).float().sum(dim=1).mean()

        # Compute average top-1 and top-5 accuracies
        top1_accuracy = (top1_accuracy_txt2img + top1_accuracy_img2txt) * 0.5 * 100
        topk_accuracy = (topk_accuracy_txt2img + topk_accuracy_img2txt) * 0.5 * 100

        return contrastive_loss.float(), top1_accuracy, topk_accuracy
    
    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
        """
        assert (
            self.lang_model._use_cached_vision_x or vision_x is not None
        ), "Must provide either vision_x or have precached media using cache_media()."

        if self.lang_model._use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when media has been cached using cache_media(). Try uncache_media() first."
            assert self.lang_model.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            vision_embeds = self._encode_vision_x(vision_x=vision_x) # [B,T,N,D] -< [B,T,D](first token)
            self._condition_media_locations(input_ids=lang_x)
        output = self.lang_model(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            output_hidden_states=True
        )
        if clear_conditioned_layers:
            self.lang_model.clear_conditioned_layers()

        unimodal_text_embeds = output.hidden_states[self.uni_modal_layers-1] # coca is -1 for cls token, bert is 0 for cls token
        text_latents = repeat(self.text_latent, "n d -> b t n d", b=unimodal_text_embeds.shape[0], t=1)
        text_latents = self.text_learn_attnetion_layer(unimodal_text_embeds.unsqueeze(1), text_latents) + text_latents
        text_latents = self.text_learn_ff(text_latents) + text_latents # 32 x 1 x 1 x 2048
        text_embeds = self.text_cls_norm(text_latents[:, 0, 0])
        # print(vision_embeds.shape, text_embeds.shape)
        contrastive_loss, top1_ce_accuracy, top5_ce_accuracy = self._compute_contrastive_loss(vision_embeds, text_embeds) # average over all frames and all chunks after resampler
        # if self.use_text_memory:
        #     self.text_memory.add(torch.randn(2, 512, 2, 64))  # (batch, seq, key | value, feature dim)
        #     key_values, mask = self.text_memory.search(torch.randn(2, 512, 64), topk = 32)
        
        return output, contrastive_loss, top1_ce_accuracy, top5_ce_accuracy

    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        num_beams=1,
        max_new_tokens=None,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        no_repeat_ngram_size=0,
        prefix_allowed_tokens_fn=None,
        length_penalty=1.0,
        num_return_sequences=1,
        do_sample=False,
        early_stopping=False,
    ):
        """
        Generate text conditioned on vision and language inputs.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                images in the same chunk are collated along T_img, and frames are collated along F
                currently only F=1 is supported (single-frame videos)
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            max_length (int, optional): Maximum length of the output. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            num_beams (int, optional): Number of beams. Defaults to 1.
            max_new_tokens (int, optional): Maximum new tokens. Defaults to None.
            temperature (float, optional): Temperature. Defaults to 1.0.
            top_k (int, optional): Top k. Defaults to 0.
            top_p (float, optional): Top p. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): No repeat ngram size. Defaults to 0.
            length_penalty (float, optional): Length penalty. Defaults to 1.0.
            num_return_sequences (int, optional): Number of return sequences. Defaults to 1.
            do_sample (bool, optional): Do sample. Defaults to False.
            early_stopping (bool, optional): Early stopping. Defaults to False.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)

        self.lang_model._use_cached_vision_x = True
        self._encode_vision_x(vision_x=vision_x)

        output = self.lang_model.generate(
            lang_x,
            attention_mask=attention_mask,
            eos_token_id=self.eoc_token_id,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            early_stopping=early_stopping,
            pad_token_id=self.lang_model.config.pad_token_id,
        )

        self.lang_model.clear_conditioned_layers()
        self.lang_model._use_cached_vision_x = False
        return output
    

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                T_img: interlevel image number
                Images in the same chunk are collated along T_img, and frames are collated along F
                F=1 (single-frame video)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        if self.vision_encoder_name == "sparseformer":
            # print("1", vision_x.shape)
            vision_x = rearrange(vision_x, "b T F c h w -> (b T) F c h w")
            vision_x = vision_x.permute(0, 2, 1, 3, 4) # b, c, temporal, h, w
            # print("2", vision_x.shape)
            vision_embeds, vision_tokens = self.vision_encoder.visual(vision_x) # report error RuntimeError: expected scalar type Half but found BFloat16
            vision_tokens = vision_tokens.permute(1, 0, 2) # from [64*inflation, batch_size, 1024] to [batch_size, 64*inflation, 1024]
            vision_tokens = rearrange(vision_tokens, "(b T) v d -> b T v d", b=b, T=T)
            vision_embeds = rearrange(vision_embeds, "(b T) d -> b T d", b=b, T=T)
            vision_embeds = torch.mean(vision_embeds[:], dim=[1])
            # print("3", vision_embeds.shape, vision_tokens.shape)
        else:
            vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
            with torch.no_grad():
                vision_x = self.vision_encoder.visual(vision_x)[1] # report error RuntimeError: expected scalar type Half but found BFloat16
            vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F) #  10 x 3 x 1 x 256 x 1024
            vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d) 10 x 3 x 64 x 1024
            vision_embeds, vision_tokens = vision_x[:, :, 0, :].unsqueeze(2), vision_x[:, :, 1:, :]  # take the first token as the cls token
            vision_embeds = torch.mean(vision_embeds[:], dim=[1,2])
        for layer in self.lang_model._get_decoder_layers():
            layer.condition_vis_x(vision_tokens)
        return vision_embeds
    
    def get_visual_text_embedding(self, vision_x: torch.Tensor, lang_x: torch.Tensor, attention_mask: torch.Tensor = None, labels: torch.Tensor = None, past_key_values=None):
        """
        compute the visual-text embedding for retrieval task
        """
        vision_embeds = self._encode_vision_x(vision_x=vision_x)
        output = self.lang_model(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values
        )
        unimodal_text_embeds = output.hidden_states[self.uni_modal_layers-1] # coca is -1 for cls token, bert is 0 for cls token
        text_latents = repeat(self.text_latent, "n d -> b t n d", b=unimodal_text_embeds.shape[0], t=1)
        text_latents = self.text_learn_attnetion_layer(unimodal_text_embeds.unsqueeze(1), text_latents) + text_latents
        text_latents = self.text_learn_ff(text_latents) + text_latents
        text_embeds = self.text_cls_norm(text_latents[:, 0, 0])
        text_latents = self.text_to_latents(text_embeds)
        visual_latents = self.img_to_latents(vision_embeds)
        return visual_latents, text_latents

    def _condition_media_locations(self, input_ids: torch.Tensor):
        """
        Compute the media token locations from lang_x and condition the language model on these.
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt)
        """
        media_locations = input_ids == self.media_token_id

        for layer in self.lang_model._get_decoder_layers():
            layer.condition_media_locations(media_locations)

    def cache_media(self, input_ids: torch.Tensor, vision_x: torch.Tensor):
        """
        Pre-cache a prompt/sequence of images / text for log-likelihood evaluations.
        All subsequent calls to forward() will generate attending to the LAST
        image in vision_x.
        This is not meant to be used to cache things for generate().
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt)
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)
        """
        self._encode_vision_x(vision_x=vision_x)
        self._condition_media_locations(input_ids=input_ids)
        self.lang_model._use_cached_vision_x = True

    def uncache_media(self):
        """
        Clear all conditioning.
        """
        self.lang_model.clear_conditioned_layers()
        self.lang_model._use_cached_vision_x = False

def gather_tensors_all_nodes(tensor):
    """
    We find this function works well for single node, but not for multi-node
    So we want to modify this function to gathered for gpus on same node
    """
    gathered_tensors = torch.cat(torch.distributed.nn.all_gather(tensor), dim=0)
    return gathered_tensors

"""
This implementation is from Ziteng Gao
"""

def group_all_gather(tensor, group, group_size, group_rank=-1, dim=-1):
    return GroupAllGather.apply(tensor, dim, group, group_size, group_rank)

class GroupAllGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, tensor: torch.Tensor, dim: int, group, group_size: int, group_rank: int
    ):
        if group_rank == -1:
            assert group is not None
            if hasattr(dist, "get_group_rank"):
                group_rank = dist.get_group_rank(group, dist.get_rank())
            else:
                group_rank = dist.get_rank(group)
        ctx.group_rank = group_rank
        ctx.group_size = group_size
        ctx.dim = dim
        ctx.group = group
        tensor_list = [torch.empty_like(tensor) for _ in range(group_size)]
        dist.all_gather(tensor_list, tensor, group=group)
        gathered = torch.cat(tensor_list, dim=dim)
        return gathered

    @staticmethod
    def backward(ctx, gathered_grad: torch.Tensor):
        group_rank = ctx.group_rank
        group_size = ctx.group_size
        dim = ctx.dim
        group = ctx.group
        gathered_grad = gathered_grad / group_size
        grad_list = list(gathered_grad.chunk(group_size, dim))
        grad_tensor = torch.empty_like(grad_list[group_rank])  # placeholder
        dist.reduce_scatter(grad_tensor, grad_list, group=group)
        return grad_tensor, None, None, None, None