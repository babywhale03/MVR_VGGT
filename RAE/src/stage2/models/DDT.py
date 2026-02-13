from math import sqrt
from re import L
from regex import B
import torch
import torch.nn as nn
import sys

from transformers import SwinModel
import torch
from torch import nn
from .lightningDiT import PatchEmbed, Mlp, NormAttention
from timm.models.vision_transformer import PatchEmbed, Mlp
from .model_utils import VisionRotaryEmbeddingFast, RMSNorm, SwiGLUFFN, GaussianFourierEmbedding, LabelEmbedder, NormAttention, get_2d_sincos_pos_embed
import torch.nn.functional as F
from typing import *

vggt_path = "/mnt/dataset1/jaeeun/MVR/vggt"
if vggt_path not in sys.path:
    sys.path.append(vggt_path)

from vggt.layers.block import Block
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
        use_qknorm=False,
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
        self.attn = NormAttention(
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

    def forward(self, x, c, feat_rope=None):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)  # (B, 1, C)
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(6, dim=-1)
        x = x + DDTGate(self.attn(DDTModulate(self.norm1(x),
                        shift_msa, scale_msa), rope=feat_rope), gate_msa)
        x = x + DDTGate(self.mlp(DDTModulate(self.norm2(x),
                        shift_mlp, scale_mlp)), gate_mlp)
        return x


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
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DDTModulate(self.norm_final(x), shift, scale)  # no gate
        x = self.linear(x)
        return x

class MultiViewDDTBlock(nn.Module):
    """
    Multiview DDT Block with frame and global attention while maintaining AdaLN modulation.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=False,
        **block_kwargs
    ):
        super().__init__()
        
        self.frame_block = LightningDDTBlock(
            hidden_size, num_heads, mlp_ratio,
            use_qknorm, use_swiglu, use_rmsnorm, wo_shift, **block_kwargs
        )
        
        self.global_block = LightningDDTBlock(
            hidden_size, num_heads, mlp_ratio,
            use_qknorm, use_swiglu, use_rmsnorm, wo_shift, **block_kwargs
        )
    
    def forward(self, x, c, num_views=1, feat_rope=None):
        """
        Args:
            x: [B*S, P, C] where S is number of views
            c: [B*S, C] or [B*S, 1, C] - AdaLN conditioning
            num_views: number of views S
            feat_rope: rotary position embedding
        """
        B_times_S, P, C = x.shape
        B = B_times_S // num_views
        
        # Frame attention: process each view independently
        # Keep shape as [B*S, P, C]
        x = self.frame_block(x, c, feat_rope=feat_rope)
        
        # Global attention: process across all views
        # Reshape to [B, S*P, C] for cross-view attention
        x_global = x.view(B, num_views * P, C)
        
        # Reshape c for global attention if needed
        if len(c.shape) == 2:
            c_global = c.view(B, num_views, C).mean(dim=1, keepdim=True)  # [B, 1, C]
        else:
            c_global = c.view(B, num_views, -1, C).mean(dim=1)  # [B, 1, C] or [B, N, C]
        
        # Apply global attention
        x_global = self.global_block(x_global, c_global, feat_rope=None)  # No RoPE for global
        
        # Reshape back to [B*S, P, C]
        x = x_global.view(B_times_S, P, C)
        
        return x

class DiTwDDTHead(nn.Module):
    def __init__(
            self,
            input_size: int = 1,
            patch_size: Union[list, int] = 1,
            in_channels: int = 768,
            hidden_size=[1152, 2048],
            depth=[28, 2],
            num_heads: Union[list[int], int] = [16, 16],
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            wo_shift=False,
            use_pos_embed: bool = True,
            use_multiview_attention=False,
            aa_order=["frame", "global"],
            aa_block_size=1,
            rope_freq=100,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels

        self.encoder_hidden_size = hidden_size[0]
        self.decoder_hidden_size = hidden_size[1]
        self.num_heads = [num_heads, num_heads] if isinstance(
            num_heads, int) else list(num_heads)
        self.num_decoder_blocks = depth[1]
        self.num_encoder_blocks = depth[0]
        self.num_blocks = depth[0] + depth[1]
        self.use_rope = use_rope
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
        self.x_embedder = PatchEmbed(
            x_input_size, x_patch_size, x_channel_per_token, self.decoder_hidden_size, bias=True)
        self.s_embedder = PatchEmbed(
            s_input_size, s_patch_size, s_channel_per_token, self.encoder_hidden_size, bias=True)
        self.s_channel_per_token = s_channel_per_token
        self.x_channel_per_token = x_channel_per_token
        self.s_projector = nn.Linear(
            self.encoder_hidden_size, self.decoder_hidden_size) if self.encoder_hidden_size != self.decoder_hidden_size else nn.Identity()
        self.t_embedder = GaussianFourierEmbedding(self.encoder_hidden_size)
        self.y_embedder = LabelEmbedder(
            num_classes, self.encoder_hidden_size, class_dropout_prob)
        # print(f"x_channel_per_token: {x_channel_per_token}, s_channel_per_token: {s_channel_per_token}")
        self.final_layer = DDTFinalLayer(
            self.decoder_hidden_size, 1, x_channel_per_token, use_rmsnorm=use_rmsnorm)
        # Will use fixed sin-cos embedding:
        if use_pos_embed:
            # num_patches = self.s_embedder.num_patches
            num_patches = 1041
            self.pos_embed = nn.Parameter(torch.zeros(
                1, num_patches, self.encoder_hidden_size), requires_grad=False)
            self.x_pos_embed = None
        self.use_pos_embed = use_pos_embed
        enc_num_heads = self.num_heads[0]
        dec_num_heads = self.num_heads[1]
        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            enc_half_head_dim = self.encoder_hidden_size // enc_num_heads // 2
            hw_seq_len = int(sqrt(1041))
            # print(f"enc_half_head_dim: {enc_half_head_dim}, hw_seq_len: {hw_seq_len}")
            self.enc_feat_rope = VisionRotaryEmbeddingFast(
                dim=enc_half_head_dim,
                pt_seq_len=hw_seq_len,
            )
            dec_half_head_dim = self.decoder_hidden_size // dec_num_heads // 2
            hw_seq_len = int(sqrt(self.x_embedder.num_patches))
            # print(f"dec_half_head_dim: {dec_half_head_dim}, hw_seq_len: {hw_seq_len}")
            self.dec_feat_rope = VisionRotaryEmbeddingFast(
                dim=dec_half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None

        self.initialize_weights()
        self.use_multiview_attention = use_multiview_attention
        self.aa_order = aa_order
        self.aa_block_size = aa_block_size
        
        if use_multiview_attention:
            # Create separate frame and global blocks for encoder
            self.enc_frame_blocks = nn.ModuleList([
                LightningDDTBlock(self.encoder_hidden_size, enc_num_heads,
                                 mlp_ratio=mlp_ratio, use_qknorm=use_qknorm,
                                 use_rmsnorm=use_rmsnorm, use_swiglu=use_swiglu,
                                 wo_shift=wo_shift) 
                for _ in range(self.num_encoder_blocks)
            ])
            self.enc_global_blocks = nn.ModuleList([
                LightningDDTBlock(self.encoder_hidden_size, enc_num_heads,
                                 mlp_ratio=mlp_ratio, use_qknorm=use_qknorm,
                                 use_rmsnorm=use_rmsnorm, use_swiglu=use_swiglu,
                                 wo_shift=wo_shift)
                for _ in range(self.num_encoder_blocks)
            ])
            
            # Create separate frame and global blocks for decoder
            self.dec_frame_blocks = nn.ModuleList([
                LightningDDTBlock(self.decoder_hidden_size, dec_num_heads,
                                 mlp_ratio=mlp_ratio, use_qknorm=use_qknorm,
                                 use_rmsnorm=use_rmsnorm, use_swiglu=use_swiglu,
                                 wo_shift=wo_shift)
                for _ in range(self.num_decoder_blocks)
            ])
            self.dec_global_blocks = nn.ModuleList([
                LightningDDTBlock(self.decoder_hidden_size, dec_num_heads,
                                 mlp_ratio=mlp_ratio, use_qknorm=use_qknorm,
                                 use_rmsnorm=use_rmsnorm, use_swiglu=use_swiglu,
                                 wo_shift=wo_shift)
                for _ in range(self.num_decoder_blocks)
            ])
        else:
            self.blocks = nn.ModuleList([
                LightningDDTBlock(self.encoder_hidden_size if i < self.num_encoder_blocks else self.decoder_hidden_size,
                                  enc_num_heads if i < self.num_encoder_blocks else dec_num_heads,
                                  mlp_ratio=mlp_ratio, use_qknorm=use_qknorm,
                                  use_rmsnorm=use_rmsnorm, use_swiglu=use_swiglu,
                                  wo_shift=wo_shift)
                for i in range(self.num_blocks)
            ])
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None


    def _process_encoder_multiview(self, s, c, num_views):
        """Process encoder with alternating frame/global attention"""
        B_times_S, P, C_dim = s.shape
        B = B_times_S // num_views
        
        frame_idx = 0
        global_idx = 0
        
        for layer_num in range(self.num_encoder_blocks):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    # Frame attention: [B*S, P, C]
                    s = self.enc_frame_blocks[frame_idx](s, c, feat_rope=self.enc_feat_rope)
                    frame_idx += 1
                elif attn_type == "global":
                    # Global attention: [B, S*P, C]
                    s_global = s.view(B, num_views * P, C_dim)
                    c_global = c.view(B, num_views, -1).mean(dim=1, keepdim=True) if len(c.shape) == 2 else c.view(B, num_views, -1, C_dim).mean(dim=1)
                    s_global = self.enc_global_blocks[global_idx](s_global, c_global, feat_rope=None)
                    s = s_global.view(B_times_S, P, C_dim)
                    global_idx += 1
        
        return s
    
    def _process_decoder_multiview(self, x, s, num_views):
        """Process decoder with alternating frame/global attention"""
        B_times_S, P, C_dim = x.shape
        B = B_times_S // num_views
        
        frame_idx = 0
        global_idx = 0
        
        for layer_num in range(self.num_decoder_blocks):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    # Frame attention: [B*S, P, C]
                    x = self.dec_frame_blocks[frame_idx](x, s, feat_rope=self.dec_feat_rope)
                    frame_idx += 1
                elif attn_type == "global":
                    # Global attention: [B, S*P, C]
                    x_global = x.view(B, num_views * P, C_dim)
                    # s is conditioning, keep same shape or aggregate
                    x_global = self.dec_global_blocks[global_idx](x_global, s, feat_rope=None)
                    x = x_global.view(B_times_S, P, C_dim)
                    global_idx += 1
        
        return x

    def initialize_weights(self, xavier_uniform_init: bool = False):
        if xavier_uniform_init:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        if self.use_pos_embed:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.s_embedder.num_patches ** 0.5))
            self.pos_embed.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0))

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

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        # c = self.out_channels
        c = self.x_channel_per_token
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, s=None, mask=None, num_views=1):
        # x: [32, 1041, 4096]
        # x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t) # t: [32, ] -> [32, 4096]
        # y = self.y_embedder(y, self.training)
        c = nn.functional.silu(t)
        if s is None:
            # s = self.s_embedder(x)
            B, P, C = x.shape
            if self.use_pos_embed:
                if self.rope is not None:
                    pos = self.position_getter(B, P, C, device=x.device)
                # TODO: 여기 작성하다 말음
                s = s + self.pos_embed
            # print(f"t shape: {t.shape}, y shape: {y.shape}, c shape: {c.shape}, s shape: {s.shape}, pos_embed shape: {self.pos_embed.shape}")
            # for i in range(self.num_encoder_blocks):
            #    s = self.blocks[i](s, c, feat_rope=self.enc_feat_rope)
            # broadcast t to s

        if self.use_multiview_attention and num_views > 1:
            # Expand c to match [B*S, C]
            if len(c.shape) == 2 and c.shape[0] != s.shape[0]:
                c = c.unsqueeze(1).repeat(1, num_views, 1).view(-1, c.shape[-1])
        
            s = self._process_encoder_multiview(s, c, num_views)
        else:
            for i in range(self.num_encoder_blocks):
                s = self.blocks[i](s, c, feat_rope=self.enc_feat_rope)
        
        t = t.unsqueeze(1).repeat(1, s.shape[1], 1)
        s = nn.functional.silu(t + s)
        s = self.s_projector(s)

        # Decoder Input
        x = self.x_embedder(x)
        if self.use_pos_embed and self.x_pos_embed is not None:
            x = x + self.x_pos_embed

        if self.use_multiview_attention and num_views > 1:
            x = self._process_decoder_multiview(x, s, num_views)
        else:
            for i in range(self.num_encoder_blocks, self.num_blocks):
                x = self.blocks[i](x, s, feat_rope=self.dec_feat_rope)
        
        x = self.final_layer(x, s)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=(0, 1)):
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
