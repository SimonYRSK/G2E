import torch
from torch import nn
from torch.nn import functional as F
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage

from typing import Sequence, Type

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


    
class UTransformer(nn.Module):
    """原始UNet版本的Transformer（无VAE）"""
    def __init__(self, embed_dim, num_groups, input_resolution, num_heads, window_size, depth):
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
        
        x_concat = torch.cat([shortcut, x_swin], dim=1)
        x_up = self.up(x_concat)
        
        return x_up  # 仅返回特征，无mu/log_var


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


if __name__ == "__main__":
    B = 1
    inchans = outchans = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input = torch.randn(B, inchans, 2,721, 1440).to(device)
    print(f"input: {input.shape}")
    model = G2EBase().to(device)

    output = model(input)

    print(f"output: {output.shape}")