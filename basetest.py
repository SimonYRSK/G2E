import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage

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
        print(f"After proj: {x.shape}")
        

        if self.norm is not None:
            # LayerNorm 需要 (B, H', W', C) 格式
            x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        
        
        
        return x



class PatchMerge(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels * 2
         
        self.reduction = nn.Linear(in_channels * 4, self.out_channels, bias=False)
        self.norm = norm_layer(in_channels * 4) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0
        
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, H // 2, W // 2, C * 4)
        
        x = self.norm(x)
        x = self.reduction(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        print(f"PatchMerge output: {x.shape}")
        return x


class PatchExpansion(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels // 2
        
        self.expansion = nn.Linear(in_channels, self.out_channels * 4, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C % 2 == 0
        
        x = x.permute(0, 2, 3, 1)                       # (B, H, W, C)
        x = self.expansion(x)                           # (B, H, W, out*4)
        x = x.view(B, H, W, self.out_channels, 2, 2)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(B, self.out_channels, H * 2, W * 2)
        x = x.contiguous()
        
        print(f"PatchExpansion output: {x.shape}")
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

class Encoder(nn.Module):
    def __init__(self, dim, num_groups, num_stages, output_reso, swin_depth, window_size, num_heads, using_checkpoints = True):
        super().__init__()
        self.down = nn.ModuleList()
        self.res = nn.ModuleList()

        input_reso = int(output_reso[0] // (2**num_stages)), int(output_reso[1] // (2**num_stages))
        input_reso = list(input_reso)

        for i in range(num_stages):
            self.down.append(Downblock(dim, dim))
            self.res.append(ResBlock(num_groups, dim))

        self.using_checkpoints = using_checkpoints



        self.swin = SwinTransformerV2Stage(
            dim= dim,
            out_dim = dim,
            window_size=window_size,
            depth=swin_depth,
            output_nchw = True,
            input_resolution=input_reso,  # 固定分辨率！
            num_heads= num_heads,           
            
        )

    def forward(self, x):
        for down, res in zip(self.down, self.res):
            x = down(x)
            x = res(x)

        x = x.permute(0, 2, 3, 1)

        if self.using_checkpoints:
            self.swin.grad_checkpointing = True

        x = self.swin(x)
       

        return x


class Decoder(nn.Module):
    def __init__(self, dim, num_groups, num_stages, output_reso, swin_depth, window_size, num_heads, using_checkpoints = True):
        super().__init__()
        self.up = nn.ModuleList()
        self.res = nn.ModuleList()
        out_size = list(output_reso)
        input_reso = int(output_reso[0] // (2**num_stages)), int(output_reso[1] // (2**num_stages))
        input_reso = list(input_reso)

        self.using_checkpoints = using_checkpoints

        for i in range(num_stages):
            self.up.append(Upblock(dim, dim, out_size))
            self.res.append(ResBlock(num_groups, dim))

        self.swin = SwinTransformerV2Stage(
            dim= dim,
            out_dim = dim,
            window_size=window_size,
            depth=swin_depth,
            output_nchw = True,
            input_resolution=input_reso,  # 固定分辨率！
            num_heads= num_heads,           
            
        )


    def forward(self, x):
        x = x.permute(0, 2, 3, 1)

        if self.using_checkpoints:
            self.swin.grad_checkpointing = True
            
        x = self.swin(x)


        for res, up in zip(self.res, self.up):
            x = res(x)
            x = up(x)

        return x




class VAE(nn.Module):
    def __init__(self, dim, num_groups, num_stages, output_reso, swin_depth, window_size, num_heads, using_checkpoints = True, **kwarg):
        super().__init__()

        window_size = to_2tuple(window_size)
       

        self.latent_dim = dim
        
        self.encoder = Encoder(
            dim,
            num_groups,
            num_stages,
            output_reso,
            swin_depth,
            window_size, 
            num_heads,
            using_checkpoints = True,

        )
        
        self.mu_proj = nn.Conv2d(dim, self.latent_dim, kernel_size=1)
        self.log_var_proj = nn.Conv2d(dim, self.latent_dim, kernel_size=1)
        self.latent2feat = nn.Conv2d(self.latent_dim, dim, kernel_size=1)
        
        self.decoder = Decoder(
            dim,
            num_groups,
            num_stages,
            output_reso,
            swin_depth,
            window_size, 
            num_heads,
            using_checkpoints = True,

        )


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, x):
        x = self.encoder(x)

        mu = self.mu_proj(x)
        log_var = self.log_var_proj(x)
        z = self.reparameterize(mu, log_var)
        z = self.latent2feat(z)

        z = self.decoder(z)
        return z, mu, log_var
  

class G2E(nn.Module):
    def __init__(
        self,
        img_size=(721, 1440),
        patch_size=(4, 4),
        in_chans=10,
        embed_dim=1536,
        num_groups=32,
        num_heads=8,
        num_stages=2,
        window_size=9,
        depth = 12,
        latent_dim = 1536,
        **kwargs

    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = in_chans
        self.patch_size = patch_size
        self.img_size = img_size
        input_resolution = int(img_size[0] / patch_size[0]), int(img_size[1] / patch_size[1])

        self.patch_emb = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        
        self.mid_layer = VAE(
            embed_dim,
            num_groups,
            num_stages,
            input_resolution,
            depth,
            window_size, 
            num_heads,
        )
        
        self.fc = nn.Linear(embed_dim, self.out_chans * 4 * 4)
        

    def forward(self, x):
        print(f"Input: {x.shape}")
        
        
        x = self.patch_emb(x)

        
       
        x, mu, log_var = self.mid_layer(x)


        #... to do ...     
        
        # 补齐奇数维度（721×1440）
        x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        
        return x


# 测试代码
if __name__ == "__main__":
    device = "cuda"
    print(f"using:{device}")
    
    x = torch.randn(2, 10, 721, 1440).to(device)
    
    model = G2E(
        img_size=(721, 1440),
        patch_size=(4, 4),
        in_chans=10,
        embed_dim=1536,  
    ).to(device)

    out = model(x)
