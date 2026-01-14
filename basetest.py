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

# ========== 基础模块库 ==========

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

class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.fc(x)

class FrequencyModule(nn.Module):
    def __init__(self, ch, freq_ratio=0.5):
        super().__init__()
        self.freq_ratio = freq_ratio
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2d(x)
        h_cut = int(H * self.freq_ratio)
        w_cut = int(W * self.freq_ratio + 1)
        x_fft[:, :, h_cut:, w_cut:] *= 0.5
        x_ifft = torch.fft.irfft2d(x_fft, s=(H, W))
        return x_ifft

class SpatialTransformer(nn.Module):
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
        h = F.avg_pool2d(h, kernel_size=self.stride, stride=self.stride)
        _, _, hs, ws = h.shape
        h = h.flatten(2).transpose(1, 2)
        h = self.encoder(h)
        h = h.transpose(1, 2).reshape(B, C, hs, ws)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        h = self.proj_out(h)
        return h + x

# ========== 模块工厂 ==========

class ModuleFactory:
    """根据字符串名称创建模块"""
    
    @staticmethod
    def create_block(block_type: str, ch: int, norm="gn", act="silu", **kwargs):
        if block_type == "resblock":
            return ResBlock(ch, norm, act)
        elif block_type == "channel_attn":
            return ChannelAttention(ch, reduction=kwargs.get("reduction", 16))
        elif block_type == "freq":
            return FrequencyModule(ch, freq_ratio=kwargs.get("freq_ratio", 0.5))
        elif block_type == "transformer":
            return SpatialTransformer(
                ch,
                nhead=kwargs.get("nhead", 4),
                stride=kwargs.get("stride", 4),
                ff_mult=kwargs.get("ff_mult", 4),
                depth=kwargs.get("depth", 1)
            )
        else:
            raise ValueError(f"Unknown block type: {block_type}")
    
    @staticmethod
    def create_downsample(in_ch: int, out_ch: int, down_type="conv", **kwargs):
        if down_type == "conv":
            return nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        elif down_type == "maxpool":
            return nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_ch, out_ch, 1)
            )
        else:
            raise ValueError(f"Unknown down_type: {down_type}")
    
    @staticmethod
    def create_upsample(in_ch: int, out_ch: int, up_type="deconv", **kwargs):
        if up_type == "deconv":
            return nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        elif up_type == "upsample":
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch, out_ch, 1)
            )
        else:
            raise ValueError(f"Unknown up_type: {up_type}")

# ========== 完全可插拔 Encoder ==========

class FlexibleEncoder(nn.Module):
    """
    完全可插拔的编码器：
    encoder_cfg = {
        "stage0": {
            "blocks": ["resblock", "resblock", "channel_attn"],
            "down": "conv"
        },
        "stage1": {
            "blocks": ["resblock", "transformer"],
            "down": "conv"
        },
        ...
    }
    """
    
    def __init__(self, in_ch, widths, encoder_cfg, norm="gn", act="silu", **block_kwargs):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, widths[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        for i in range(len(widths)):
            ch = widths[i]
            stage_key = f"stage{i}"
            
            # 获取该 stage 的配置（如无则用默认）
            stage_cfg = encoder_cfg.get(stage_key, {"blocks": ["resblock"], "down": "conv"})
            blocks_list = stage_cfg.get("blocks", ["resblock"])
            down_type = stage_cfg.get("down", "conv")
            
            # 构建该 stage 的块
            blocks = nn.ModuleList()
            for block_name in blocks_list:
                if block_name == "resblock":
                    blocks.append(ResBlock(ch, norm, act))
                else:
                    block = ModuleFactory.create_block(block_name, ch, norm, act, **block_kwargs)
                    blocks.append(block)
            
            self.enc_blocks.append(nn.Sequential(*blocks))
            
            # 下采样（最后一层除外）
            if i < len(widths) - 1:
                down = ModuleFactory.create_downsample(ch, widths[i+1], down_type, **block_kwargs)
                self.downs.append(down)
        
        self.downs.append(None)

    def forward(self, x):
        feats = []
        h = self.stem(x)
        for i, block in enumerate(self.enc_blocks):
            h = block(h)
            feats.append(h)
            if self.downs[i] is not None:
                h = self.downs[i](h)
        return h, feats

# ========== 完全可插拔 Decoder ==========

class FlexibleDecoder(nn.Module):
    """
    完全可插拔的解码器：
    decoder_cfg = {
        "stage0": {
            "blocks": ["resblock", "resblock"],
            "up": "deconv"
        },
        "stage1": {
            "blocks": ["resblock", "transformer"],
            "up": "upsample"
        },
        ...
    }
    """
    
    def __init__(self, out_ch, widths, decoder_cfg, norm="gn", act="silu", **block_kwargs):
        super().__init__()
        self.up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        
        for i in range(len(widths) - 1, 0, -1):
            in_ch = widths[i]
            skip_ch = widths[i - 1]
            stage_key = f"stage{i-1}"
            
            # 获取该 stage 的配置
            stage_cfg = decoder_cfg.get(stage_key, {"blocks": ["resblock"], "up": "deconv"})
            blocks_list = stage_cfg.get("blocks", ["resblock"])
            up_type = stage_cfg.get("up", "deconv")
            
            # 上采样
            up = ModuleFactory.create_upsample(in_ch, skip_ch, up_type, **block_kwargs)
            self.up.append(up)
            
            # 拼接后的通道压缩 + 块堆叠
            blocks = [
                nn.Conv2d(skip_ch * 2, skip_ch, kernel_size=1),
                make_norm(norm, skip_ch),
                make_act(act),
            ]
            
            for block_name in blocks_list:
                if block_name == "resblock":
                    blocks.append(ResBlock(skip_ch, norm, act))
                else:
                    block = ModuleFactory.create_block(block_name, skip_ch, norm, act, **block_kwargs)
                    blocks.append(block)
            
            self.dec_blocks.append(nn.Sequential(*blocks))
        
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
        return self.head(h)

# ========== 完全可插拔 VAE ==========

class VAEUNetFlex(nn.Module):
    def __init__(
        self,
        in_ch=10,
        out_ch=10,
        widths=(64, 128, 256),
        encoder_cfg=None,
        decoder_cfg=None,
        norm="gn",
        act="silu",
        latent_dim=None,
        **block_kwargs
    ):
        super().__init__()
        self.widths = widths
        self.latent_dim = latent_dim or widths[-1]
        
        # 默认配置
        if encoder_cfg is None:
            encoder_cfg = {
                f"stage{i}": {"blocks": ["resblock"], "down": "conv"}
                for i in range(len(widths))
            }
        
        if decoder_cfg is None:
            decoder_cfg = {
                f"stage{i}": {"blocks": ["resblock"], "up": "deconv"}
                for i in range(len(widths) - 1)
            }
        
        self.encoder = FlexibleEncoder(in_ch, widths, encoder_cfg, norm, act, **block_kwargs)
        
        bott_ch = widths[-1]
        self.mu = nn.Conv2d(bott_ch, self.latent_dim, 1)
        self.logvar = nn.Conv2d(bott_ch, self.latent_dim, 1)
        nn.init.constant_(self.logvar.weight, -5.0)
        nn.init.constant_(self.logvar.bias, -5.0)
        
        self.lat2feat = nn.Conv2d(self.latent_dim, bott_ch, 1)
        self.decoder = FlexibleDecoder(out_ch, widths, decoder_cfg, norm, act, **block_kwargs)

    @staticmethod
    def _pad_to_multiple(x, mult=4):
        _, _, h, w = x.shape
        pad_h = (mult - h % mult) % mult
        pad_w = (mult - w % mult) % mult
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (h, w)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, -10.0, 10.0)
        std = torch.exp(0.5 * logvar) + 1e-6
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(2)
        
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

    # ========== 方案 1: 基础 Baseline ==========
    model1 = VAEUNetFlex(
        in_ch=C, out_ch=C,
        widths=(64, 128, 256),
        encoder_cfg={
            "stage0": {"blocks": ["resblock", "resblock"], "down": "conv"},
            "stage1": {"blocks": ["resblock", "resblock"], "down": "conv"},
            "stage2": {"blocks": ["resblock", "resblock"], "down": "conv"},
        },
        decoder_cfg={
            "stage0": {"blocks": ["resblock", "resblock"], "up": "deconv"},
            "stage1": {"blocks": ["resblock", "resblock"], "up": "deconv"},
        },
    ).to(device)

    # ========== 方案 2: 加入注意力 ==========
    model2 = VAEUNetFlex(
        in_ch=C, out_ch=C,
        widths=(64, 128, 256),
        encoder_cfg={
            "stage0": {"blocks": ["resblock", "resblock"], "down": "conv"},
            "stage1": {"blocks": ["resblock", "transformer"], "down": "conv"},
            "stage2": {"blocks": ["resblock", "transformer"], "down": "conv"},
        },
        decoder_cfg={
            "stage0": {"blocks": ["resblock", "resblock"], "up": "deconv"},
            "stage1": {"blocks": ["resblock", "transformer"], "up": "deconv"},
        },
        nhead=4, stride=4, ff_mult=4, depth=1,
    ).to(device)

    # ========== 方案 3: 混合模块 ==========
    model3 = VAEUNetFlex(
        in_ch=C, out_ch=C,
        widths=(64, 128, 256),
        encoder_cfg={
            "stage0": {"blocks": ["resblock", "channel_attn"], "down": "conv"},
            "stage1": {"blocks": ["resblock", "freq", "transformer"], "down": "maxpool"},
            "stage2": {"blocks": ["resblock", "channel_attn"], "down": "conv"},
        },
        decoder_cfg={
            "stage0": {"blocks": ["resblock", "channel_attn"], "up": "upsample"},
            "stage1": {"blocks": ["resblock", "freq"], "up": "deconv"},
        },
        nhead=8, stride=4, ff_mult=4, depth=1,
    ).to(device)

    # ========== 方案 4: 深层模型 ==========
    model4 = VAEUNetFlex(
        in_ch=C, out_ch=C,
        widths=(64, 128, 256, 512),
        encoder_cfg={
            "stage0": {"blocks": ["resblock", "resblock"], "down": "conv"},
            "stage1": {"blocks": ["resblock", "resblock", "channel_attn"], "down": "conv"},
            "stage2": {"blocks": ["resblock", "resblock", "transformer"], "down": "conv"},
            "stage3": {"blocks": ["resblock", "transformer"], "down": "conv"},
        },
        decoder_cfg={
            "stage0": {"blocks": ["resblock", "resblock"], "up": "deconv"},
            "stage1": {"blocks": ["resblock", "transformer"], "up": "deconv"},
            "stage2": {"blocks": ["resblock", "channel_attn"], "up": "upsample"},
        },
        nhead=8, stride=4, ff_mult=4, depth=2,
    ).to(device)

    for i, model in enumerate([model1, model2, model3, model4], 1):
        y, mu, logvar = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"Model {i}: output={y.shape}, params={params/1e6:.2f}M")