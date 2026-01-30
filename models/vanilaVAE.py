from .base import *
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F


class VanillaEncoder(Encoder):
    """
    去掉 Swin 的純卷積編碼器
    """
    def __init__(self, dim, num_groups, num_stages, using_checkpoints=True):
        # 只傳必要的參數，刻意不傳 swin 相關的
        super().__init__(
            dim=dim,
            num_groups=num_groups,
            num_stages=num_stages,
            output_reso=(1, 1),           # 占位，後面不會用到
            swin_depth=0,                 # 故意設為 0
            window_size=7,
            num_heads=8,
            using_checkpoints=using_checkpoints
        )
        # 關鍵：移除或替換 swin 模塊
        self.swin = nn.Identity()


    def forward(self, x):
        # 不做 permute → swin → permute
        for down, res in zip(self.down, self.res):
            x = down(x)
            if self.using_checkpoints:
                x = checkpoint.checkpoint(res, x, use_reentrant=False)
            else:
                x = res(x)
        return x


class VanillaDecoder(Decoder):
    """
    去掉 Swin 的純卷積解碼器
    """
    def __init__(self, dim, num_groups, num_stages, output_reso, using_checkpoints=True):
        super().__init__(
            dim=dim,
            num_groups=num_groups,
            num_stages=num_stages,
            output_reso=output_reso,
            swin_depth=0,
            window_size=7,
            num_heads=8,
            using_checkpoints=using_checkpoints
        )
        self.swin = nn.Identity()


    def forward(self, x):
        # 不做 permute → swin
        for res, up in zip(self.res, self.up):
            if self.using_checkpoints:
                x = checkpoint.checkpoint(res, x, use_reentrant=False)
            else:
                x = res(x)
            x = up(x)
        return x


class VanillaVAE(VAE):
    """
    使用純卷積編解碼器的 VAE（無 Swin）
    """
    def __init__(
        self,
        dim,
        num_groups=32,
        num_stages=2,
        output_reso=(180, 360),          # patch 後的解析度，例如 721//4, 1440//4
        using_checkpoints=True,
        **kwargs
    ):
        # 傳入必要的參數，但 swin 相關設為無效值
        super().__init__(
            dim=dim,
            num_groups=num_groups,
            num_stages=num_stages,
            output_reso=output_reso,
            swin_depth=0,               # 故意無效
            window_size=7,
            num_heads=8,
            using_checkpoints=using_checkpoints,
            **kwargs
        )

        # 覆蓋掉原本帶 Swin 的 encoder / decoder
        self.encoder = VanillaEncoder(
            dim=dim,
            num_groups=num_groups,
            num_stages=num_stages,
            using_checkpoints=using_checkpoints
        )

        self.decoder = VanillaDecoder(
            dim=dim,
            num_groups=num_groups,
            num_stages=num_stages,
            output_reso=output_reso,
            using_checkpoints=using_checkpoints
        )



class G2Esimple(G2E):
    """
    無 Swin 版本，只把 mid_layer 換成 VanillaVAE
    """
    def __init__(
        self,
        img_size=(721, 1440),
        patch_size=(4, 4),
        in_chans=70,
        out_chans=None,
        embed_dim=1536,               # 建議先用較小的 dim 測試
        num_groups=32,
        num_stages=2,
        using_checkpoints=True,
        **kwargs
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dim=embed_dim,
            num_groups=num_groups,
            num_heads=8,             # 雖然不用但保持接口
            num_stages=num_stages,
            window_size=7,
            depth=0,                 # 無意義
            latent_dim=embed_dim,
            **kwargs
        )

        # 替換成不帶 Swin 的 VAE
        self.mid_layer = VanillaVAE(
            dim=embed_dim,
            num_groups=num_groups,
            num_stages=num_stages,
            output_reso=(img_size[0] // patch_size[0], img_size[1] // patch_size[1]),
            using_checkpoints=using_checkpoints
        )

