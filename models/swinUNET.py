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


def time_to_features(timestamp: str, height: int, width: int) -> np.ndarray:
    """将单个时间戳编码为 4 个时间特征通道，并 broadcast 到 (H, W)。

    month/day 的正余弦编码，格式参照用户提供的示例函数。
    返回形状为 (4, H, W) 的 numpy 数组，dtype=float32。
    """
    dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")
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
    def __init__(self, num_groups, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, ch)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, ch)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
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
    ):
        super().__init__()
        self.num_stages = num_stages
        self.using_checkpoints = using_checkpoints

        window_size = to_2tuple(window_size)

        base_h, base_w = output_reso
        self.stage_resolutions = []
        for i in range(num_stages):
            h_i = int(base_h // (2 ** i))
            w_i = int(base_w // (2 ** i))
            self.stage_resolutions.append((h_i, w_i))

        # 每个 stage 的 ResBlock 数量
        if res_per_stage is None:
            res_per_stage = [1] * num_stages
        elif isinstance(res_per_stage, int):
            res_per_stage = [res_per_stage] * num_stages
        else:
            assert len(res_per_stage) == num_stages, "len(res_per_stage) 必须等于 num_stages"
        self.res_per_stage = res_per_stage

        # 将总 depth 均分到各个 Swin stage，至少为 1
        depth_per_stage = max(1, swin_depth // max(1, num_stages))

        # 每个 stage 一组 ResBlock
        self.res_blocks = nn.ModuleList()  # List[ModuleList[ResBlock]]
        self.swin_stages = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        for i in range(num_stages):
            stage_res_blocks = nn.ModuleList(
                [ResBlock(num_groups, dim) for _ in range(self.res_per_stage[i])]
            )
            self.res_blocks.append(stage_res_blocks)

            input_reso = self.stage_resolutions[i]
            swin = SwinTransformerV2Stage(
                dim=dim,
                out_dim=dim,
                window_size=window_size,
                depth=depth_per_stage,
                output_nchw=True,
                input_resolution=input_reso,
                num_heads=num_heads,
            )
            if using_checkpoints:
                swin.grad_checkpointing = True
            self.swin_stages.append(swin)

            # 最后一层不再下采样
            if i < num_stages - 1:
                self.down_blocks.append(Downblock(dim, dim))

    def forward(self, x):
        skips = []
        h = x
        for i in range(self.num_stages):
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

    def __init__(self, dim, num_groups, num_stages, output_reso, using_checkpoints: bool = True):
        super().__init__()
        self.num_stages = num_stages
        self.using_checkpoints = using_checkpoints

        base_h, base_w = output_reso
        self.stage_resolutions = []
        for i in range(num_stages):
            h_i = int(base_h // (2 ** i))
            w_i = int(base_w // (2 ** i))
            self.stage_resolutions.append((h_i, w_i))

        # 上采样层数 = 下采样层数 = num_stages - 1
        self.up_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        for idx in range(num_stages - 1):
            # 从分辨率 stage i+1 上采样到 stage i
            i = num_stages - 2 - idx  # 反向遍历：先从最小尺度往上
            out_size = self.stage_resolutions[i]
            self.up_blocks.append(Upblock(dim, dim, out_size))
            # 拼接 skip 后通道数为 2*dim
            self.res_blocks.append(ResBlock(num_groups, dim * 2))

    def forward(self, x, skips):
        # skips: 长度 num_stages，其中最后一个是 bottleneck 尺度
        h = x
        for idx, (up, res) in enumerate(zip(self.up_blocks, self.res_blocks)):
            i = self.num_stages - 2 - idx
            h = up(h)
            h = torch.cat([h, skips[i]], dim=1)
            if self.using_checkpoints:
                h = checkpoint.checkpoint(res, h, use_reentrant=False)
            else:
                h = res(h)
        return h


class UNet(nn.Module):
    """基于 PatchEmbedding 的单个 Swin-UNet（非 VAE）。

    Patch 之后进入多尺度 Encoder：每个 stage CNN + Swin + Down，
    bottleneck 之后通过 Decoder 逐级上采样并融合 skip。
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
        **kwargs,
    ):
        super().__init__()
        window_size = to_2tuple(window_size)

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
        )

        self.decoder = UNetDecoder(
            dim,
            num_groups,
            num_stages,
            output_reso,
            using_checkpoints=using_checkpoints,
        )

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
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
        **kwargs

    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = in_chans if out_chans is None else out_chans
        self.patch_size = patch_size
        self.img_size = img_size
        self.using_checkpoints = using_checkpoints
        self.using_time_embedding = using_time_embedding
        # 若使用时间嵌入，则在输入通道上多拼接 4 个时间特征通道
        self.time_channels = 4 if using_time_embedding else 0
        input_resolution = int(img_size[0] / patch_size[0]), int(img_size[1] / patch_size[1])

        # Patch 之前就将时间特征拼到像素通道，故这里的 in_chans 需要加上 time_channels
        self.patch_emb = PatchEmbedding(img_size, patch_size, in_chans + self.time_channels, embed_dim)

        # 中间层改为 UNet（非 VAE）
        self.mid_layer = UNet(
            embed_dim,
            num_groups,
            num_stages,
            input_resolution,
            depth,
            window_size, 
            num_heads,
            using_checkpoints,
            res_per_stage,
        )
        
        self.patch_head = PatchHead(embed_dim, self.out_chans, patch_size)
        

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

        # UNet 中间层，不再是 VAE，没有采样与 KL 项
        x = self.mid_layer(x_patch)

        if self.using_checkpoints:
            x = checkpoint.checkpoint(self.patch_head, x, use_reentrant=False)
        else:
            x = self.patch_head(x)

        x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        return x

# 测试代码

# 测试代码
if __name__ == "__main__":
    device = "cuda"
    print(f"using: {device}")
    
    # 打印初始显存
    print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    x = torch.randn(13, 10, 721, 1440).to(device)
    target = torch.randn(13, 10, 721, 1440).to(device)  # 目标数据
    
    print(f"After data load GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    model = G2E(
        img_size=(721, 1440),
        patch_size=(4, 4),
        in_chans=10,
        embed_dim=1536,  
    ).to(device)
    
    print(f"After model load GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # ========== 训练逻辑 ==========
    model.train()  # 切换到训练模式（checkpoint 只在 train 模式生效）
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    from torch.cuda.amp import autocast, GradScaler
    scaler = torch.cuda.amp.GradScaler()
    num_epochs = 3
    
    for epoch in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        print(f"Before forward GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, target)

        print(f"After forward GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB, loss={loss.item():.6f}")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 清理与统计
        torch.cuda.synchronize()
        print(f"After step GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

        # 清理缓存并重置 peak 统计（有助于下一 epoch 内存分配）
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    print("\n========== Training completed ==========")
    print(f"Final GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
