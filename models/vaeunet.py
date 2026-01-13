import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 工厂函数 ==========
def make_norm(norm: str, ch: int):
    if norm == "bn":
        return nn.BatchNorm2d(ch)
    if norm == "in":
        return nn.InstanceNorm2d(ch, affine=True)
    if norm == "gn":
        return nn.GroupNorm(8, ch)
    raise ValueError(f"unknown norm {norm}")

def make_act(act: str):
    return {"relu": nn.ReLU(inplace=True), "silu": nn.SiLU(), "gelu": nn.GELU()}.get(act, nn.ReLU(inplace=True))

# ========== 基础模块 ==========
class ResBlock(nn.Module):
    def __init__(self, ch, norm="gn", act="silu"):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = make_norm(norm, ch)
        self.act = make_act(act)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = make_norm(norm, ch)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + x)

class SpatialTransformer(nn.Module):
    """
    轻量 Transformer：先空间降采样（stride），再做 EncoderLayer，再还原尺寸。
    注意：tokens = (H/stride)*(W/stride)，请用小 stride 控制开销。
    """
    def __init__(self, ch, nhead=4, stride=4, ff_mult=4, depth=1):
        super().__init__()
        self.stride = stride
        self.proj_in = nn.Conv2d(ch, ch, 1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=ch,
            nhead=nhead,
            dim_feedforward=ch * ff_mult,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.proj_out = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.proj_in(x)
        h = F.avg_pool2d(h, kernel_size=self.stride, stride=self.stride)  # 下采样
        _, _, hs, ws = h.shape
        h = h.flatten(2).transpose(1, 2)  # (B, N, C)
        h = self.encoder(h)
        h = h.transpose(1, 2).reshape(B, C, hs, ws)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        h = self.proj_out(h)
        return h + x  # 残差

# ========== VAE 编码/解码 ==========
class Encoder(nn.Module):
    def __init__(self, in_ch, widths, res_blocks_per_stage, norm="gn", act="silu",
                 use_attn=False, attn_cfg=None):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, widths[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.attn = nn.ModuleList()

        for i in range(len(widths)):
            ch = widths[i]
            # 堆叠 ResBlocks
            stage = nn.Sequential(*[ResBlock(ch, norm, act) for _ in range(res_blocks_per_stage[i])])
            self.enc_blocks.append(stage)
            # 可选注意力
            if use_attn and attn_cfg is not None and attn_cfg.get("stages", [False]*len(widths))[i]:
                self.attn.append(
                    SpatialTransformer(
                        ch,
                        nhead=attn_cfg.get("nhead", 4),
                        stride=attn_cfg.get("stride", 4),
                        ff_mult=attn_cfg.get("ff_mult", 4),
                        depth=attn_cfg.get("depth", 1),
                    )
                )
            else:
                self.attn.append(None)
            # 下采样（最后一层不下采样）
            if i < len(widths) - 1:
                self.downs.append(nn.Conv2d(ch, widths[i + 1], 3, stride=2, padding=1))
        self.downs.append(None)  # 占位，便于索引

    def forward(self, x):
        feats = []
        h = self.stem(x)
        for i, block in enumerate(self.enc_blocks):
            h = block(h)
            if self.attn[i] is not None:
                h = self.attn[i](h)
            feats.append(h)
            if self.downs[i] is not None:
                h = self.downs[i](h)
        return h, feats  # h: bottleneck, feats: skip

# ...existing code...
class Decoder(nn.Module):
    def __init__(self, out_ch, widths, res_blocks_per_stage, norm="gn", act="silu",
                 use_attn=False, attn_cfg=None):
        super().__init__()
        self.up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.attn = nn.ModuleList()

        for i in range(len(widths) - 1, 0, -1):
            in_ch = widths[i]
            skip_ch = widths[i - 1]
            self.up.append(nn.ConvTranspose2d(in_ch, skip_ch, 2, stride=2))
            # 先 1x1 将 2*skip_ch 压到 skip_ch，再堆叠 ResBlock(skip_ch)
            blocks = [
                nn.Conv2d(skip_ch * 2, skip_ch, kernel_size=1),
                make_norm(norm, skip_ch),
                make_act(act),
            ] + [ResBlock(skip_ch, norm, act) for _ in range(res_blocks_per_stage[i - 1])]
            self.dec_blocks.append(nn.Sequential(*blocks))

            if use_attn and attn_cfg is not None and attn_cfg.get("stages", [False]*len(widths))[i - 1]:
                self.attn.append(
                    SpatialTransformer(
                        skip_ch,
                        nhead=attn_cfg.get("nhead", 4),
                        stride=attn_cfg.get("stride", 4),
                        ff_mult=attn_cfg.get("ff_mult", 4),
                        depth=attn_cfg.get("depth", 1),
                    )
                )
            else:
                self.attn.append(None)
        self.head = nn.Conv2d(widths[0], out_ch, 1)

    def forward(self, h, feats):
        for i, up in enumerate(self.up):
            h = up(h)
            skip = feats[-(i + 2)]
            if h.shape[2:] != skip.shape[2:]:
                H = min(h.shape[2], skip.shape[2])
                W = min(h.shape[3], skip.shape[3])
                h = h[:, :, :H, :W]
                skip = skip[:, :, :H, :W]
            h = torch.cat([h, skip], dim=1)
            h = self.dec_blocks[i](h)
            if self.attn[i] is not None:
                h = self.attn[i](h)
        return self.head(h)
# ...existing code...
# ========== 顶层 VAE ==========
class VAEUNet(nn.Module):
    """
    可插拔 VAE UNet：
    - widths 控制通道/深度，如 (64,128,256)
    - res_blocks_per_stage 与 widths 同长，控制每层 ResBlock 数
    - use_attn + attn_cfg 控制可选 Transformer 模块
    """
    def __init__(
        self,
        in_ch=10,
        out_ch=10,
        widths=(64, 128, 256),
        res_blocks_per_stage=(1, 1, 1),
        norm="gn",
        act="silu",
        use_attn=False,
        attn_cfg=None,
        latent_dim=None,
    ):
        super().__init__()
        assert len(widths) == len(res_blocks_per_stage)
        self.widths = widths
        self.latent_dim = latent_dim or widths[-1]

        self.encoder = Encoder(in_ch, widths, res_blocks_per_stage, norm, act, use_attn, attn_cfg)
        bott_ch = widths[-1]
        self.mu = nn.Conv2d(bott_ch, self.latent_dim, 1)
        self.logvar = nn.Conv2d(bott_ch, self.latent_dim, 1)
        self.lat2feat = nn.Conv2d(self.latent_dim, bott_ch, 1)

        self.decoder = Decoder(out_ch, widths, res_blocks_per_stage, norm, act, use_attn, attn_cfg)

    @staticmethod
    def _pad_to_multiple(x, mult=4):
        _, _, h, w = x.shape
        pad_h = (mult - h % mult) % mult
        pad_w = (mult - w % mult) % mult
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (h, w)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 兼容 (B, C, T, H, W) 或 (B, C, H, W)
        if x.dim() == 5:
            if x.shape[2] != 1:
                raise ValueError(f"期待时间维 T=1，当前 T={x.shape[2]}")
            x = x.squeeze(2)
        elif x.dim() != 4:
            raise ValueError(f"输入维度应为 4 或 5，当前 {x.shape}")
        # x: (B, C, H=721, W=1440)  上游请先 squeeze 时间维
        x, orig_hw = self._pad_to_multiple(x, mult=4)
        h, feats = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        h = self.lat2feat(z)
        out = self.decoder(h, feats)
        H, W = orig_hw
        out = out[:, :, :H, :W]
        out = F.interpolate(out, size=(721, 1440), mode="bilinear", align_corners=False)
        return out, mu, logvar

if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C = 2, 10
    x = torch.randn(B, C, 1, 721, 1440, device=device).squeeze(2)

    model = VAEUNet(
        in_ch=C,
        out_ch=C,
        widths=(64, 128, 256),
        res_blocks_per_stage=(4, 4, 4),
        use_attn=True,
        attn_cfg={"stages": [False, True, True], "nhead": 8, "stride": 4, "ff_mult": 4, "depth": 1},
    ).to(device)

    y, mu, logvar = model(x)
    print("input :", x.shape)
    print("output:", y.shape, "mu:", mu.shape, "logvar:", logvar.shape)