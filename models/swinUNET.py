import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint 
import pandas as pd
import numpy as np
from datetime import datetime
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage
from einops import rearrange

class ModuleFactory:
    def create_block(dim, out_dim, depth, input_resolution, window_size, **kwargs):
        
        
        return SwinTransformerV2Stage(
            dim= dim,
            out_dim = out_dim,
            window_size=window_size,
            depth=depth,
            input_resolution=input_resolution,  # 固定分辨率！
            num_heads=kwargs.get("num_heads", 8),           
            use_checkpoint=kwargs.get("use_checkpoint", False)
        )

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
    """
    Args:
        input_resolution (tuple[int]): Lat, Lon
        window_size (tuple[int]): Lat, Lon

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom)
    """
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[: 4]


def time_to_features(timestamp, height: int, width: int) -> np.ndarray:
    """将单个时间戳编码为 4 个时间特征通道，并 broadcast 到 (H, W)。

    支持多种常见格式，例如 "YYYY-MM-DD HH:MM:SS"、"YYYY-MM-DDTHH:MM:SS" 等，
    内部统一用 pandas.Timestamp 解析。
    返回形状为 (4, H, W) 的 numpy 数组，dtype=float32。
    """
    # 统一转成字符串再交给 pandas 解析，更健壮
    dt = pd.Timestamp(str(timestamp))
    month = dt.month
    day = dt.day
    hour = dt.hour  # 目前未使用，但保留以便后续扩展

    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    day_sin = np.sin(2 * np.pi * day / 31.0)
    day_cos = np.cos(2 * np.pi * day / 31.0)

    time_features = np.array(
        [month_sin, month_cos, day_sin, day_cos], dtype=np.float32
    ).reshape(4, 1, 1)
    time_features = np.broadcast_to(time_features, (4, height, width))
    return time_features


def time_to_features_batch(timestamps, height: int, width: int, device: torch.device) -> torch.Tensor:
    """将一批字符串时间戳编码为 [B, 4, H, W] 的时间特征张量。"""
    if isinstance(timestamps, (list, tuple)):
        ts_list = list(timestamps)
    else:
        # 允许传入一维 numpy / tensor，统一转成 python list 的字符串
        ts_list = [str(t) for t in timestamps]

    features = [time_to_features(ts, height, width) for ts in ts_list]
    features = np.stack(features, axis=0)  # [B, 4, H, W]
    return torch.from_numpy(features).to(device)




class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(721, 1440), patch_size=(4, 4), in_chans=4, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        
        # 注意：这里 img_size[2] 会报错，因为 img_size 是 2D 元组，只有 2 个元素
        # 修正：patches_resolution 只用前两个维度
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.norm = norm_layer(embed_dim) if norm_layer else None
        self.patches_resolution = patches_resolution  # 保存分辨率用于后续 reshape

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input size ({H}×{W}) doesn't match model ({self.img_size[0]}×{self.img_size[1]})"
        
        x = self.proj(x)                        # (B, embed_dim, H', W')
        
        

        if self.norm is not None:
            # LayerNorm 需要 (B, H', W', C) 格式
            x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        
        
        
        return x


class ResBlock(nn.Module):
    def __init__(self, num_groups, ch, dropout_rate: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, ch)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate and dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, ch)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        return self.act(h + x)
        

class Downblock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size = 3, stride = 2, padding = 1)
      
    def forward(self, x):
        
        x = self.conv(x)
        return x

class Upblock(nn.Module):
    def __init__(self, in_chans, out_chans, out_size):
        super().__init__()
        self.size = out_size
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):

        return self.conv(F.interpolate(x, size = tuple(self.size), mode = "bilinear"))

    

class UNetEncoder(nn.Module):
    """Patch 之后的多尺度 Encoder：每个 stage 先若干个 CNN(ResBlock)，再 Swin，然后下采样。

    输出 bottleneck 特征和各尺度的 skip list，用于解码端融合。
    """

    def __init__(
        self,
        dim,
        num_groups,
        num_stages,
        output_reso,
        swin_depth,
        window_size,
        num_heads,
        using_checkpoints: bool = True,
        res_per_stage=None,
        dims=None,
        dropout_rate: float = 0.0,
        use_residual_blocks: bool = True,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.using_checkpoints = using_checkpoints
        self.use_residual_blocks = use_residual_blocks

        window_size = to_2tuple(window_size)

        base_h, base_w = output_reso
        self.stage_resolutions = []
        for i in range(num_stages):
            h_i = int(base_h // (2 ** i))
            w_i = int(base_w // (2 ** i))
            self.stage_resolutions.append((h_i, w_i))

        # 每个 stage 的通道数配置：
        # - 若 dims 为 None，则所有 stage 使用相同的 dim
        # - 若提供 list/tuple，则长度必须等于 num_stages
        if dims is None:
            dims = [dim] * num_stages
        else:
            assert len(dims) == num_stages, "len(dims) 必须等于 num_stages"
            dims = list(dims)
        self.dims = dims

        # 每个 stage 的 ResBlock 数量
        if res_per_stage is None:
            res_per_stage = [1] * num_stages
        elif isinstance(res_per_stage, int):
            res_per_stage = [res_per_stage] * num_stages
        else:
            assert len(res_per_stage) == num_stages, "len(res_per_stage) 必须等于 num_stages"
        self.res_per_stage = res_per_stage

        # Swin depth 配置：
        # - 若传入 int，则均分到各个 stage，且每个 stage 至少 1 层
        # - 若传入 list/tuple，则逐 stage 指定，可为 0 表示该 stage 不使用 Swin
        if isinstance(swin_depth, int):
            depth_per_stage = [max(1, swin_depth // max(1, num_stages))] * num_stages
        else:
            assert len(swin_depth) == num_stages, "len(swin_depth) 必须等于 num_stages"
            depth_per_stage = list(swin_depth)
        self.depth_per_stage = depth_per_stage

        # 每个 stage 一组 ResBlock
        self.res_blocks = nn.ModuleList()  # List[ModuleList[ResBlock]]
        self.swin_stages = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        for i in range(num_stages):
            ch = self.dims[i]
            if self.use_residual_blocks:
                stage_res_blocks = nn.ModuleList(
                    [ResBlock(num_groups, ch, dropout_rate=dropout_rate) for _ in range(self.res_per_stage[i])]
                )
            else:
                stage_res_blocks = nn.ModuleList([])
            self.res_blocks.append(stage_res_blocks)

            input_reso = self.stage_resolutions[i]
            d_i = self.depth_per_stage[i]
            if d_i > 0:
                swin = SwinTransformerV2Stage(
                    dim=ch,
                    out_dim=ch,
                    window_size=window_size,
                    depth=d_i,
                    output_nchw=True,
                    input_resolution=input_reso,
                    num_heads=num_heads,
                )
                if using_checkpoints:
                    swin.grad_checkpointing = True
            else:
                # depth 为 0 时，该 stage 不使用 Swin，直接用恒等映射占位
                swin = nn.Identity()
            self.swin_stages.append(swin)

            # 最后一层不再下采样
            if i < num_stages - 1:
                self.down_blocks.append(Downblock(self.dims[i], self.dims[i + 1]))

    def forward(self, x):
        skips = []
        h = x
        for i in range(self.num_stages):
            ch = self.dims[i]
            # 当前 stage 内的若干个 ResBlock 串联
            for rb in self.res_blocks[i]:
                if self.using_checkpoints:
                    h = checkpoint.checkpoint(rb, h, use_reentrant=False)
                else:
                    h = rb(h)

            # Swin block，按当前分辨率
            h_nhwc = h.permute(0, 2, 3, 1)
            h_nhwc = self.swin_stages[i](h_nhwc)
            h = h_nhwc.permute(0, 3, 1, 2)

            # 作为该尺度 skip
            skips.append(h)

            # 下采样到下一尺度
            if i < self.num_stages - 1:
                h = self.down_blocks[i](h)

        # h: bottleneck 特征；skips: 各尺度特征（含 bottleneck）
        return h, skips


class UNetDecoder(nn.Module):
    """UNet 解码端：逐级上采样并与 encoder skip 拼接，再经 CNN。"""

    def __init__(
        self,
        dim,
        num_groups,
        num_stages,
        output_reso,
        using_checkpoints: bool = True,
        dims=None,
        dropout_rate: float = 0.0,
        use_skip_connections: bool = True,
        use_residual_blocks: bool = True,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.using_checkpoints = using_checkpoints
        self.use_skip_connections = use_skip_connections
        self.use_residual_blocks = use_residual_blocks

        base_h, base_w = output_reso
        self.stage_resolutions = []
        for i in range(num_stages):
            h_i = int(base_h // (2 ** i))
            w_i = int(base_w // (2 ** i))
            self.stage_resolutions.append((h_i, w_i))

        # 与 Encoder 保持一致的每个 stage 通道数
        if dims is None:
            dims = [dim] * num_stages
        else:
            assert len(dims) == num_stages, "len(dims) 必须等于 num_stages"
            dims = list(dims)
        self.dims = dims

        # 上采样层数 = 下采样层数 = num_stages - 1
        self.up_blocks = nn.ModuleList()
        # 先用一个 reduce block 将 concat 后的 2*dim 通道压回 dim，再用 ResBlock(dim)
        self.reduce_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        for idx in range(num_stages - 1):
            # 从分辨率 stage i+1 上采样到 stage i
            i = num_stages - 2 - idx  # 反向遍历：先从最小尺度往上
            out_size = self.stage_resolutions[i]
            in_ch = self.dims[i + 1]
            out_ch = self.dims[i]
            self.up_blocks.append(Upblock(in_ch, out_ch, out_size))

            # concat 之后通道数为 2*dim，先用一个 Conv+GN+SiLU 压回 dim 通道
            reduce_in_ch = out_ch * 2 if self.use_skip_connections else out_ch
            reduce_layers = [
                nn.Conv2d(reduce_in_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, out_ch),
                nn.SiLU(),
            ]
            if dropout_rate and dropout_rate > 0:
                reduce_layers.append(nn.Dropout2d(dropout_rate))
            self.reduce_blocks.append(nn.Sequential(*reduce_layers))
            # 再接一个 ResBlock(dim)
            if self.use_residual_blocks:
                self.res_blocks.append(ResBlock(num_groups, out_ch, dropout_rate=dropout_rate))
            else:
                self.res_blocks.append(nn.Identity())

    def forward(self, x, skips):
        # skips: 长度 num_stages，其中最后一个是 bottleneck 尺度
        h = x
        for idx, (up, reduce, res) in enumerate(zip(self.up_blocks, self.reduce_blocks, self.res_blocks)):
            i = self.num_stages - 2 - idx
            h = up(h)
            if self.use_skip_connections and skips is not None:
                h = torch.cat([h, skips[i]], dim=1)  # [B, 2*dim, H, W]
            h = reduce(h)  # [B, dim, H, W]
            if self.using_checkpoints:
                h = checkpoint.checkpoint(res, h, use_reentrant=False)
            else:
                h = res(h)
        return h


class UNet(nn.Module):
    """基于 PatchEmbedding 的单个 Swin-UNet，可选瓶颈 VAE（KL 重参数化）。

    Patch 之后进入多尺度 Encoder：每个 stage CNN + Swin + Down，
    若 using_kl=False：bottleneck 直接送入 Decoder；
    若 using_kl=True：在 bottleneck 处学习 (mu, log_var)，重参数化得到 z 再送 Decoder，
    并返回 (out, mu, log_var)。
    """

    def __init__(
        self,
        dim,
        num_groups,
        num_stages,
        output_reso,
        swin_depth,
        window_size,
        num_heads,
        using_checkpoints: bool = True,
        res_per_stage=None,
        dims=None,
        using_kl: bool = False,
        dropout_rate: float = 0.0,
        use_skip_connections: bool = True,
        use_residual_blocks: bool = True,
        **kwargs,
    ):
        super().__init__()
        window_size = to_2tuple(window_size)

        self.using_kl = using_kl

        # dims 为每个 stage 的通道列表；若未提供，则所有 stage 使用相同的 dim
        if dims is None:
            dims = [dim] * num_stages
        else:
            assert len(dims) == num_stages, "len(dims) 必须等于 num_stages"
            dims = list(dims)

        self.encoder = UNetEncoder(
            dim,
            num_groups,
            num_stages,
            output_reso,
            swin_depth,
            window_size,
            num_heads,
            using_checkpoints=using_checkpoints,
            res_per_stage=res_per_stage,
            dims=dims,
            dropout_rate=dropout_rate,
            use_residual_blocks=use_residual_blocks,
        )

        self.decoder = UNetDecoder(
            dim,
            num_groups,
            num_stages,
            output_reso,
            using_checkpoints=using_checkpoints,
            dims=dims,
            dropout_rate=dropout_rate,
            use_skip_connections=use_skip_connections,
            use_residual_blocks=use_residual_blocks,
        )

        # 瓶颈 VAE 的 mu / log_var 头，只在 using_kl 时创建
        if self.using_kl:
            bottleneck_ch = dims[-1]
            self.mu_head = nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1)
            self.logvar_head = nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1)

    def forward(self, x):
        bottleneck, skips = self.encoder(x)

        if self.using_kl:
            mu = self.mu_head(bottleneck)
            log_var = self.logvar_head(bottleneck)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
            out = self.decoder(z, skips)
            return out, mu, log_var
        else:
            out = self.decoder(bottleneck, skips)
            return out
    


class PatchHead(nn.Module):
    def __init__(self, embed_dim, out_chans, patch_size=(4,4)):
        super().__init__()

        self.patch_size = patch_size
        self.out_chans = out_chans
        self.head = nn.Linear(embed_dim, out_chans * patch_size[0] * patch_size[1])


    def forward(self, x):
        B, C, H, W = x.shape

        feat_h, feat_w = H, W

        x = x.flatten(2).transpose(1, 2)
        
        x = self.head(x)           # (B, H, W, out_chans * patch_size * patch_size)
        x = rearrange(
            x,
            'n (h w) (p1 p2 c) -> n c (h p1) (w p2)',
            h=feat_h,   # 180
            w=feat_w,   # 360
            p1=self.patch_size[0],    # 4
            p2=self.patch_size[1],    # 4
            c=self.out_chans       # 70
        )
        # 输出: (B, out_chans, H*p1, W*p2) = (B, 70, 720, 1440)
        
        return x
  
class G2E(nn.Module):
    def __init__(
        self,
        img_size=(721, 1440),
        patch_size=(4, 4),
        in_chans=10,
        out_chans = None,
        embed_dim=1536,
        num_groups=32,
        num_heads=8,
        num_stages=3,
        window_size=9,
        depth = 12,
        latent_dim = 1536,
        using_checkpoints = True,
        using_time_embedding = False,
        res_per_stage = [1, 2, 4],
        channels=None,
        using_kl: bool = False,
        dropout_rate: float = 0.0,
        use_skip_connections: bool = True,
        use_residual_blocks: bool = True,
        **kwargs

    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = in_chans if out_chans is None else out_chans
        self.patch_size = patch_size
        self.img_size = img_size
        self.using_checkpoints = using_checkpoints
        self.using_time_embedding = using_time_embedding
        self.using_kl = using_kl
        # 若使用时间嵌入，则在输入通道上多拼接 4 个时间特征通道
        self.time_channels = 4 if using_time_embedding else 0
        input_resolution = int(img_size[0] / patch_size[0]), int(img_size[1] / patch_size[1])

        # 每个 stage 的通道列表：
        # - 若 channels 为 None，则所有 stage 使用相同的 embed_dim
        # - 若提供 list/tuple（长度 = num_stages），则：
        #     channels[0] 为 patch 后的通道数，channels[1] 为第一个 Downblock 后通道数，依此类推
        if channels is None:
            dims = [embed_dim] * num_stages
        else:
            assert len(channels) == num_stages, "len(channels) 必须等于 num_stages"
            dims = list(channels)
        self.dims = dims

        # Patch 之前就将时间特征拼到像素通道，故这里的 in_chans 需要加上 time_channels
        # PatchEmbedding 输出通道使用第一个 stage 的通道数 dims[0]
        self.patch_emb = PatchEmbedding(img_size, patch_size, in_chans + self.time_channels, self.dims[0])

        # 中间层改为 UNet，可选 VAE 瓶颈
        self.mid_layer = UNet(
            self.dims[0],
            num_groups,
            num_stages,
            input_resolution,
            depth,
            window_size, 
            num_heads,
            using_checkpoints,
            res_per_stage,
            dims=self.dims,
            using_kl=self.using_kl,
            dropout_rate=dropout_rate,
            use_skip_connections=use_skip_connections,
            use_residual_blocks=use_residual_blocks,
        )
        
        # PatchHead 的输入通道与最高分辨率的通道数相同
        self.patch_head = PatchHead(self.dims[0], self.out_chans, patch_size)
        

    def forward(self, x, times=None, i=None):  # 时间嵌入在输入第一步完成
        B, C, H, W = x.shape

        if self.using_time_embedding and times is not None:
            # times: 长度为 B 的字符串列表，例如 "2025-01-01T00:00:00"
            time_feats = time_to_features_batch(times, H, W, x.device)  # [B, 4, H, W]
            x = torch.cat([x, time_feats], dim=1)

        if self.using_checkpoints:
            x_patch = checkpoint.checkpoint(self.patch_emb, x, use_reentrant=False)
        else:
            x_patch = self.patch_emb(x)

        # UNet 中间层，可选瓶颈 VAE
        if self.using_kl:
            mid_out, mu, log_var = self.mid_layer(x_patch)
        else:
            mid_out = self.mid_layer(x_patch)

        if self.using_checkpoints:
            x = checkpoint.checkpoint(self.patch_head, mid_out, use_reentrant=False)
        else:
            x = self.patch_head(mid_out)

        x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        if self.using_kl:
            return x, mu, log_var
        else:
            return x

# 测试代码

