import torch
from torch import nn
from torch.nn import functional as F
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage

from typing import Sequence, Type

from .base import G2EBase, DownBlock, UpBlock



def get_pad3d(input_resolution, window_size):
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    return padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back

def get_pad2d(input_resolution, window_size):
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[: 4]


class SwinVAETransformer(nn.Module):
    """VAE版本的Transformer（带重参数化）"""
    def __init__(self, embed_dim, num_groups, input_resolution, num_heads, window_size, depth, latent_dim=None):
        super().__init__()
        num_groups = to_2tuple(num_groups)
        window_size = to_2tuple(window_size)
        padding = get_pad2d(input_resolution, window_size)
        padding_left, padding_right, padding_top, padding_bottom = padding
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)
        input_resolution = list(input_resolution)
        input_resolution[0] = input_resolution[0] + padding_top + padding_bottom
        input_resolution[1] = input_resolution[1] + padding_left + padding_right
        
        self.down = DownBlock(embed_dim, embed_dim, num_groups[0])
        self.layer = SwinTransformerV2Stage(embed_dim, embed_dim, input_resolution, depth, num_heads, window_size)
        self.up = UpBlock(embed_dim * 2, embed_dim, num_groups[1])
        
        self.latent_dim = latent_dim or embed_dim
        self.mu_proj = nn.Conv2d(embed_dim, self.latent_dim, kernel_size=1)
        self.log_var_proj = nn.Conv2d(embed_dim, self.latent_dim, kernel_size=1)
        self.latent2feat = nn.Conv2d(self.latent_dim, embed_dim, kernel_size=1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        B, C, Lat, Lon = x.shape
        padding_left, padding_right, padding_top, padding_bottom = self.padding
        
        x_down = self.down(x)
        shortcut = x_down
        
        x_pad = self.pad(x_down)
        x_pad = x_pad.permute(0, 2, 3, 1)
        x_swin = self.layer(x_pad)
        x_swin = x_swin.permute(0, 3, 1, 2)
        x_swin = x_swin[:, :, padding_top:-padding_bottom, padding_left:-padding_right]
        
        mu = self.mu_proj(x_swin)
        log_var = self.log_var_proj(x_swin)
        z = self.reparameterize(mu, log_var)
        
        z_feat = self.latent2feat(z)
        z_pad = self.pad(z_feat)
        z_pad = z_pad.permute(0, 2, 3, 1)
        z_swin = self.layer(z_pad)
        z_swin = z_swin.permute(0, 3, 1, 2)
        z_swin = z_swin[:, :, padding_top:-padding_bottom, padding_left:-padding_right]
        
        x_concat = torch.cat([shortcut, z_swin], dim=1)
        x_up = self.up(x_concat)
        
        return x_up, mu, log_var


class G2EVAE(G2EBase):
    def __init__(self, latent_dim=None, **kwargs):
        # 使用SwinVAETransformer，传递latent_dim参数
        kwargs["transformer_cls"] = SwinVAETransformer
        kwargs["transformer_kwargs"] = {"latent_dim": latent_dim}
        super().__init__(**kwargs)
        self.latent_dim = latent_dim or self.embed_dim

    def _forward_transformer(self, x):
        # VAE版本返回特征+mu+log_var（暂存为实例变量，供forward使用）
        x, self.mu, self.log_var = self.transformer(x)
        return x

    def forward(self, x: torch.Tensor):
        # 返回生成结果+mu+log_var
        output = self._common_forward(x)
        return output, self.mu, self.log_var


if __name__ == "__main__":
    B = 1
    inchans = outchans = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input = torch.randn(B, inchans, 2,721, 1440).to(device)
    print(f"input: {input.shape}")
    model = G2EVAE().to(device)

    output, _, _ = model(input)

    print(f"output: {output.shape}")

    