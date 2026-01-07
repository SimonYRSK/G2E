import torch
from torch import nn
from torch.nn import functional as F
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage

from typing import Sequence, Type

from models.vaebase import G2EVAE



B = 1
in_chans = out_chans = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")
input = torch.randn(B, in_chans, 2, 721, 1440).to(device)
print(f"input: {input.shape}")

# # 测试UNet版本
# fuxi_unet = FuxiUNet(embed_dim=864, num_heads=8, window_size=7).to(device)
# output_unet = fuxi_unet(input)
# print(f"\nUNet输出尺寸: {output_unet.shape}")

# 测试VAE版本
fuxi_vae = G2EVAE(embed_dim=864, num_heads=8, window_size=7, latent_dim=864).to(device)
output_vae, mu, log_var = fuxi_vae(input)
print(f"VAE输出尺寸: {output_vae.shape}")
print(f"mu shape: {mu.shape}, log_var shape: {log_var.shape}")