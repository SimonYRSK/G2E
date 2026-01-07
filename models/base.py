import torch
from torch import nn
from torch.nn import functional as F
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage

from typing import Sequence, Type

class CubeEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]

        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, T, Lat, Lon = x.shape
        assert T == self.img_size[0] and Lat == self.img_size[1] and Lon == self.img_size[2], \
            f"Input image size ({T}*{Lat}*{Lon}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x).reshape(B, self.embed_dim, -1).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, *self.patches_resolution)
        return x




class DownBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, num_groups: int, num_residuals: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=(3, 3), stride=2, padding=1)

        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.conv(x)

        shortcut = x

        x = self.b(x)

        res = x + shortcut
        if h % 2 != 0:
            res = res[:, :, :-1, :]
        if w % 2 != 0:
            res = res[:, :, :, :-1]
        return res

class UpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)

        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.conv(x)

        shortcut = x

        x = self.b(x)

        return x + shortcut




class G2EBase(nn.Module):
    def __init__(self, 
                 img_size=(2, 721, 1440), 
                 patch_size=(2, 4, 4), 
                 in_chans=10, 
                 out_chans=10,
                 embed_dim=864, 
                 num_groups=32, 
                 num_heads=8, 
                 window_size=7, 
                 depth=16,
                 transformer_cls: Type[nn.Module] = UTransformer,  # 可插拔的Transformer类
                 transformer_kwargs: dict = None):  # Transformer的额外参数
        super().__init__()
        # 通用参数
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        
        # 计算输入分辨率
        self.input_resolution = (
            int(img_size[1] / patch_size[1] / 2), 
            int(img_size[2] / patch_size[2] / 2)
        )
        
        # 通用模块
        self.cube_embedding = CubeEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.fc = nn.Linear(embed_dim, out_chans * patch_size[1] * patch_size[2])
        
        # 初始化Transformer（核心可插拔部分）
        transformer_kwargs = transformer_kwargs or {}
        self.transformer = transformer_cls(
            embed_dim=embed_dim,
            num_groups=num_groups,
            input_resolution=self.input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            depth=depth,
            **transformer_kwargs  # 传递Transformer的额外参数（如latent_dim）
        )

    def _common_forward(self, x):
        """通用forward逻辑：CubeEmbedding → 维度还原 → 插值"""
        B = x.shape[0]
        _, patch_lat, patch_lon = self.patch_size
        Lat, Lon = self.input_resolution
        Lat, Lon = Lat * 2, Lon * 2

        # 立方体嵌入
        x = self.cube_embedding(x).squeeze(2)
        
        # 调用Transformer（子类实现）
        x = self._forward_transformer(x)
        
        # 通用维度还原
        x = self.fc(x.permute(0, 2, 3, 1))
        x = x.reshape(B, Lat, Lon, patch_lat, patch_lon, self.out_chans).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, Lat * patch_lat, Lon * patch_lon, self.out_chans)
        x = x.permute(0, 3, 1, 2)
        
        # 双线性插值
        x = F.interpolate(x, size=self.img_size[1:], mode="bilinear")
        return x

    def _forward_transformer(self, x):
        """子类需实现：Transformer的forward逻辑"""
        raise NotImplementedError("子类必须实现_forward_transformer方法")