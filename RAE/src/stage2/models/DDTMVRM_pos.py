from math import sqrt
import os
from re import L
import sys
from regex import B
import torch
import torch.nn as nn

from transformers import SwinModel
import torch
from torch import nn
from .lightningDiT import PatchEmbed, Mlp, JaeAttention
from timm.models.vision_transformer import PatchEmbed, Mlp
from .model_utils import VisionRotaryEmbeddingFast, RMSNorm, SwiGLUFFN, GaussianFourierEmbedding, LabelEmbedder, NormAttention, get_2d_sincos_pos_embed_rect
import torch.nn.functional as F
from typing import *
import matplotlib.pyplot as plt
import cv2
import numpy as np

vggt_path = "/mnt/dataset1/jaeeun/MVR/vggt"
if vggt_path not in sys.path:
    sys.path.append(vggt_path)
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
    
def DDTModulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        shift: Tensor of shape (B, L, D)
        scale: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * (1 + scale) + shift, 
        with shift and scale repeated to match L_x if necessary.
    """
    # shift, scale: [B, 1, 1152]
    B, Lx, D = x.shape
    _, L, _ = shift.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        shift = shift.repeat_interleave(repeat, dim=1)
        scale = scale.repeat_interleave(repeat, dim=1)
    # apply modulation
    return x * (1 + scale) + shift


def DDTGate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        gate: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * gate, 
        with gate repeated to match L_x if necessary.
    """
    B, Lx, D = x.shape
    _, L, _ = gate.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        # print(f"gate shape: {gate.shape}, x shape: {x.shape}")
        gate = gate.repeat_interleave(repeat, dim=1)
    # apply modulation
    return x * gate

# @torch._dynamo.disable
# def save_attention(attn_map, query_y, query_x, layer_idx, save_dir, pH=28, pW=37):
#     special_tokens = 5 
#     query_idx = special_tokens + (query_y * pW + query_x)
#     batch_query_attn = attn_map[:, :, query_idx, special_tokens:]
    
#     avg_map = batch_query_attn.mean(dim=(0, 1)).view(pH, pW).detach().cpu().float().numpy()
    
#     heatmap_norm = cv2.normalize(avg_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
#     heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_VIRIDIS)
#     heatmap_resized = cv2.resize(heatmap_color, (pW * 20, pH * 20), interpolation=cv2.INTER_NEAREST)
    
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"layer{layer_idx}_q{query_y}_{query_x}.png")
#     cv2.imwrite(save_path, heatmap_resized)

class LightningDDTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including: 
    - ROPE
    - QKNorm 
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=True,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=False,
        **block_kwargs
    ):
        super().__init__()

        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        # Initialize attention layer
        self.attn = JaeAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )

        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )

        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, pos=None):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)  # (B, 1, C)
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(4, dim=-1) # multiview: [B, 1, C]
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(6, dim=-1)
        attn_output, attn_map = self.attn(DDTModulate(self.norm1(x), shift_msa, scale_msa), pos=pos, return_attn=True)
        x = x + DDTGate(attn_output, gate_msa)
        x = x + DDTGate(self.mlp(DDTModulate(self.norm2(x),
                        shift_mlp, scale_mlp)), gate_mlp)
        return x, attn_map


class DDTFinalLayer(nn.Module):
    """
    The final layer of DDT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # breakpoint()
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DDTModulate(self.norm_final(x), shift, scale)  # no gate
        x = self.linear(x)
        return x
    
class DiTwDDTHead(nn.Module):
    def __init__(
            self,
            input_size: int = 1,
            patch_size: Union[list, int] = 1,
            vggt_patch_size: int = 14,
            in_channels: int = 768,
            hidden_size=[1152, 2048],
            depth=[28, 2],
            num_heads: Union[list[int], int] = [16, 16],
            mlp_ratio=4.0,
            use_qknorm=False,
            use_swiglu=True,
            use_rmsnorm=True,
            wo_shift=False,
            use_pos_embed: bool = True,
            rope_freq=100,
            num_register_tokens=4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.vggt_patch_size = vggt_patch_size
        self.encoder_hidden_size = hidden_size[0]
        self.decoder_hidden_size = hidden_size[1]
        self.num_heads = [num_heads, num_heads] if isinstance(
            num_heads, int) else list(num_heads)
        self.num_decoder_blocks = depth[1]
        self.num_encoder_blocks = depth[0]
        self.num_blocks = depth[0] + depth[1]
        # self.use_rope = use_rope
        # analyze patch size
        if isinstance(patch_size, int) or isinstance(patch_size, float):
            patch_size = [patch_size, patch_size]  # patch size for s , x embed
        assert len(
            patch_size) == 2, f"patch size should be a list of two numbers, but got {patch_size}"
        self.patch_size = patch_size
        self.s_patch_size = patch_size[0]
        self.x_patch_size = patch_size[1]
        s_channel_per_token = in_channels * self.s_patch_size * self.s_patch_size
        s_input_size = input_size
        s_patch_size = self.s_patch_size
        x_input_size = input_size
        x_patch_size = self.x_patch_size
        x_channel_per_token = in_channels * self.x_patch_size * self.x_patch_size
        self.x_embedder = nn.Linear(self.in_channels, self.decoder_hidden_size)
        self.s_embedder = nn.Linear(self.in_channels, self.encoder_hidden_size)
        self.s_channel_per_token = s_channel_per_token
        self.x_channel_per_token = x_channel_per_token
        self.s_projector = nn.Linear(
            self.encoder_hidden_size, self.decoder_hidden_size) if self.encoder_hidden_size != self.decoder_hidden_size else nn.Identity()
        self.t_embedder = GaussianFourierEmbedding(self.encoder_hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, self.encoder_hidden_size, class_dropout_prob)
        # print(f"x_channel_per_token: {x_channel_per_token}, s_channel_per_token: {s_channel_per_token}")
        self.final_layer = DDTFinalLayer(
            self.decoder_hidden_size, 1, self.in_channels, use_rmsnorm=use_rmsnorm)
        # self.pH = pH
        # self.pW = pW
        self.patch_start_idx = 1 + num_register_tokens
        # Will use fixed sin-cos embedding:#
        # if use_pos_embed:
        #     num_patches = pH * pW + self.patch_start_idx # 1041
        #     self.pos_embed = nn.Parameter(torch.zeros(
        #         1, num_patches, self.encoder_hidden_size), requires_grad=False)
        #     self.x_pos_embed = None
        self.use_pos_embed = use_pos_embed
        enc_num_heads = self.num_heads[0]
        dec_num_heads = self.num_heads[1]
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None
        
        self.blocks = nn.ModuleList([
            LightningDDTBlock(self.encoder_hidden_size if i < self.num_encoder_blocks else self.decoder_hidden_size, # encoder_hidden_size=1152, decoder_hidden_size=2048
                              enc_num_heads if i < self.num_encoder_blocks else dec_num_heads, # num_heads = 16
                              mlp_ratio=mlp_ratio,
                              use_qknorm=use_qknorm,
                              use_rmsnorm=use_rmsnorm,
                              use_swiglu=use_swiglu,
                              wo_shift=wo_shift,
                              rope=self.rope,
                              ) for i in range(self.num_blocks) # num_blocks = enc + dec = 30
        ])
        self.frame_blocks = None
        self.global_blocks = None
        self.initialize_weights()

    def initialize_weights(self, xavier_uniform_init: bool = False):
        if xavier_uniform_init:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
        # Initialize x_embedder:
        nn.init.xavier_uniform_(self.x_embedder.weight)
        if self.x_embedder.bias is not None:
            nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize s_embedder:
        nn.init.xavier_uniform_(self.s_embedder.weight)
        if self.s_embedder.bias is not None:
            nn.init.constant_(self.s_embedder.bias, 0)

        # TODO: s_project initialization? 

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        if self.use_pos_embed:
            D = self.pos_embed.shape[-1]
            
            pos_embed = get_2d_sincos_pos_embed_rect(
                D, self.pH, self.pW,
                cls_token=True, 
                extra_tokens=self.patch_start_idx
            )
            
            self.pos_embed.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0)
            ) # TODO: [1, 1041, 1152]인지 확인

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, clean_img, x, t, step=None, experiment_dir=None, s=None, mask=None):
        # breakpoint()
        # x: [B, S, 1041, 1024]
        # x = self.x_embedder(x) + self.pos_embed
        breakpoint()
        _, _, H, W = clean_img.shape
        pH, pW = H // self.vggt_patch_size, W // self.vggt_patch_size
        # print(f"pH: {pH}, pW: {pW}")
        
        if len(x.shape) == 4:
            x = x.squeeze(1)  # [B * S, 1041, 1024]

        t = self.t_embedder(t) # [B, 1152]? 
        # y = self.y_embedder(y, self.training) # [B, 1152]
        c = nn.functional.silu(t)
        pos = None
        if self.rope is not None:
            pos = self.position_getter(x.shape[0], pH, pW, device=x.device)
        
        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(x.shape[0], self.patch_start_idx, 2).to(x.device).to(pos.dtype) # [B * S, 5, 2]
            pos = torch.cat([pos_special, pos], dim=1) # [B * S, 1041, 2]

        # attn_save_dir = f"{experiment_dir}/attn_maps/step{step}"
        # os.makedirs(attn_save_dir, exist_ok=True) 

        # encoder
        if s is None:
            s = self.s_embedder(x) # linear: [B, 1041, 1024] -> [B, 1041, 1152]
            if self.use_pos_embed:
                s = s + self.pos_embed 
            # print(f"t shape: {t.shape}, y shape: {y.shape}, c shape: {c.shape}, s shape: {s.shape}, pos_embed shape: {self.pos_embed.shape}")
            for i in range(self.num_encoder_blocks): # num_encoder_blocks=28
                s, attn_map = self.blocks[i](s, c, pos=pos)
                # attn_map = attn_map.detach().float().cpu()
                # if attn_save_dir is not None:
                #     for query in [(14, 18), (7, 9), (21, 27)]: # (y, x)
                #         save_attn_query_dir = f"{attn_save_dir}/layer{i}_query{query[0]}_{query[1]}"
                #         save_attention(attn_map, query_y=query[0], query_x=query[1], layer_idx=i, save_dir=save_attn_query_dir)
            # broadcast t to s
            t = t.unsqueeze(1).repeat(1, s.shape[1], 1)
            # if not hasattr(self, 'norm_s_merge'):
            #     self.norm_s_merge = RMSNorm(self.encoder_hidden_size).to(s.device)
            # s = nn.functional.silu(t.unsqueeze(1) + self.norm_s_merge(s))
            s = nn.functional.silu(t + s)
            # if s.dim() > 3:
            #     s = s.squeeze(1)

        s = self.s_projector(s) # linear layer [32, 1041, 2048]
        x = self.x_embedder(x) # linear: [32, 1041, 1024] -> [32, 1041, 2048]
        if self.use_pos_embed and self.x_pos_embed is not None:
            x = x + self.x_pos_embed

        # decoder
        for i in range(self.num_encoder_blocks, self.num_blocks): # num_decoder_blocks=2
            x, attn_map = self.blocks[i](x, s, pos=pos)
            # attn_map = attn_map.detach().float().cpu()
            # if attn_save_dir is not None:
            #     for query in [(14, 18), (7, 9), (21, 27)]: # (y, x)
            #         save_attn_query_dir = f"{attn_save_dir}/layer{i}_query{query[0]}_{query[1]}"
            #         save_attention(attn_map, query_y=query[0], query_x=query[1], layer_idx=i, save_dir=save_attn_query_dir)
        x = self.final_layer(x, s) 
        return x 

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=(0, 1)):
        # breakpoint()
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:,
                              :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
        t = t[: len(t) // 2] # get t for the conditional half
        half_eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)
             ).view(-1, *[1] * (len(cond_eps.shape) - 1)),
            uncond_eps + cfg_scale * (cond_eps - uncond_eps), cond_eps
        )
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward_with_autoguidance(self, x, t, y, cfg_scale, additional_model_forward, cfg_interval=(0, 1)):
        # breakpoint()
        """
        Forward pass of LightningDiT, but also contain the forward pass for the additional model
        """
        model_out = self.forward(x, t, y)
        ag_model_out = additional_model_forward(x, t, y)
        eps = model_out[:, :self.in_channels]
        ag_eps = ag_model_out[:, :self.in_channels]

        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
        eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)
             ).view(-1, *[1] * (len(eps.shape) - 1)),
            ag_eps + cfg_scale * (eps - ag_eps), eps
        )

        return eps
        
