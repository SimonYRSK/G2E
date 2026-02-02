from .base import *
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss


class DAEncoder(Encoder):  # 可以用 VanillaEncoder，如果已去 Swin
    def __init__(self, *args, using_swin=False, **kwargs):
        super().__init__(
            *args, **kwargs
        )
        self.using_swin = using_swin

        if not using_swin:
            self.swin = nn.Identity()  # 确保无 Swin

    def forward(self, x):
        for down, res in zip(self.down, self.res):
            x = down(x)
            if self.using_checkpoints:
                x = checkpoint.checkpoint(res, x, use_reentrant=False)
            else:
                x = res(x)

        if not self.using_swin:
            return x

        else :
            x = x.permute(0, 2, 3, 1)
            x = self.swin(x)
            x = x.permute(0, 3, 1, 2) 
            return x
        
class DADecoder(Decoder):  
    def __init__(self, *args, using_swin=False, **kwargs):
        super().__init__(
            *args, **kwargs
        )
        self.using_swin = using_swin

        if not using_swin:
            self.swin = nn.Identity()  # 确保无 Swin

    def forward(self, x):
        if not self.using_swin:
            for res, up in zip(self.res, self.up):
                if self.using_checkpoints:
                    x = checkpoint.checkpoint(res, x, use_reentrant=False)
                else:
                    x = res(x)
                x = up(x)
            return x

        else :
            x = x.permute(0, 2, 3, 1)
            x = self.swin(x)
            x = x.permute(0, 3, 1, 2) 

            for res, up in zip(self.res, self.up):
                if self.using_checkpoints:
                    x = checkpoint.checkpoint(res, x, use_reentrant=False)
                else:
                    x = res(x)
                x = up(x)
            return x

class DAVAE(nn.Module):
    def __init__(
        self,
        dim,
        num_groups,
        num_stages,
        output_reso,
        swin_depth,
        window_size,
        num_heads,
        using_checkpoints=True,
        soft_share=True,
        using_swin=False,
        **kwargs
    ):
        super().__init__()
        self.latent_dim = dim

        self.encoder_gfs = DAEncoder(
            dim=dim,
            num_groups=num_groups,
            num_stages=num_stages,
            output_reso=output_reso,
            using_checkpoints=using_checkpoints,
            using_swin=False  
        )

        self.encoder_era5 = DAEncoder(
            dim=dim,
            num_groups=num_groups,
            num_stages=num_stages,
            output_reso=output_reso,
            using_checkpoints=using_checkpoints,
            using_swin=False  
        )
        self.soft_share = soft_share
        if not soft_share:
            pass
        else :
            self.encoder_gfs.load_state_dict(self.encoder_era5.state_dict())

        self.mu_proj = nn.Conv2d(dim, self.latent_dim, kernel_size=1)
        self.log_var_proj = nn.Conv2d(dim, self.latent_dim, kernel_size=1)

        self.mapping = nn.Sequential(
            nn.Flatten(2),  # B,C,H*W
            nn.Linear(output_reso[0] * output_reso[1], output_reso[0] * output_reso[1]),  # 线性映射
            nn.ReLU(),
            nn.Linear(output_reso[0] * output_reso[1], output_reso[0] * output_reso[1]),
            nn.Unflatten(2, output_reso)  # 回 B,C,H,W
        )


        self.latent2feat = nn.Conv2d(self.latent_dim, dim, kernel_size=1)

        self.mmd_fn = MMD_loss()

        self.decoder = DADecoder(dim, num_groups, num_stages, output_reso, using_checkpoints)


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, gfs, era5):
        feat_gfs = self.encoder_gfs(gfs)
        feat_era5 = self.encoder_era5(era5)

        mu_gfs = self.mu_proj(feat_gfs)
        log_var_gfs = self.log_var_proj(feat_gfs)
        z_gfs = self.reparameterize(mu_gfs, log_var_gfs)

        mu_era5 = self.mu_proj(feat_era5)
        log_var_era5 = self.log_var_proj(feat_era5)
        z_era5 = self.reparameterize(mu_era5, log_var_era5)

        mapped_z = self.mapping(z_gfs)

        feat = self.latent2feat(mapped_z)
        recon_era5 = self.decoder(feat)

        return recon_era5
    
    def compute_loss(self, recon_era5, era5, mu_era5, log_var_era5, mu_gfs, log_var_gfs, mapped_z, z_era5, lambda_kl=0.001, lambda_share=0.01, lambda_align=0.1):
        recon_loss = F.mse_loss(recon_era5, era5)

        kl_era5 = -0.5 * torch.mean(1 + log_var_era5 - mu_era5.pow(2) - log_var_era5.exp())
        kl_gfs = -0.5 * torch.mean(1 + log_var_gfs - mu_gfs.pow(2) - log_var_gfs.exp())
        
        # 域对齐：MMD on mapped_z 和 z_era5 (扁平化)
        align_loss = self.mmd_fn(mapped_z.flatten(2), z_era5.flatten(2))
        
        # 软共享正则（如果启用）
        share_loss = 0
        if self.soft_share:
            for p_era5, p_gfs in zip(self.encoder_era5.parameters(), self.encoder_gfs.parameters()):
                share_loss += F.mse_loss(p_era5, p_gfs)
        
        total_loss = recon_loss + lambda_kl * (kl_era5 + kl_gfs) + lambda_align * align_loss + lambda_share * share_loss
        return recon_loss, kl_era5, kl_gfs, align_loss, share_loss, total_loss


class DAG2E(G2E):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mid_layer = DAVAE(*args, **kwargs)

    def forward(self, gfs, era5):
        # Patch embedding on both (假设共享 embedding)
        if self.training:
            gfs = checkpoint.checkpoint(self.patch_emb, gfs, use_reentrant=False)
            era5 = checkpoint.checkpoint(self.patch_emb, era5, use_reentrant=False)
        else:
            gfs = self.patch_emb(gfs)
            era5 = self.patch_emb(era5)

        recon_era5= self.mid_layer(gfs, era5)

        if self.training:
            recon_era5 = checkpoint.checkpoint(self.patch_head, recon_era5, use_reentrant=False)
        else:
            recon_era5 = self.patch_head(recon_era5)

        recon_era5 = F.interpolate(recon_era5, size=self.img_size, mode='bilinear', align_corners=False)
        return recon_era5
