import torch
import torch.nn as nn
import torch.nn.functional as F
from .swinUNET import G2E as BaseG2E
import numpy as np
import torch.utils.checkpoint as checkpoint 

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target):
        # source, target: [B, D]
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)  # [2B, D]
        total0 = total.unsqueeze(0)  # [1, 2B, D]
        total1 = total.unsqueeze(1)  # [2B, 1, D]
        L2_distance = ((total0 - total1) ** 2).sum(2)  # [2B, 2B]
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bw_temp) for bw_temp in bandwidth_list]
        return sum(kernel_val)  # [2B, 2B]

    def forward(self, source, target):
        # 支持 [B, C, H, W] 或 [B, D]
        if source.dim() > 2:
            source = source.flatten(1)
        if target.dim() > 2:
            target = target.flatten(1)
        batch_size = source.size(0)
        kernels = self.gaussian_kernel(source, target)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


class G2ELatent(BaseG2E):
    def __init__(self, *args, kernel_num=5, kernel_mul=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        # latent_loss_cfg: dict, e.g. {"mmd": 1.0, "coral": 0.5, ...}

        self.cal_mmd= MMD_loss(kernel_mul=kernel_num, kernel_num=kernel_num)

    def forward(self, x, y, times=None):
        # x: GFS, y: ERA5 (for latent loss)
        # return_latent: True 时返回 (out, latent_x, latent_y)
        # 1. GFS->ERA5 预测
        B, C, H, W = x.shape
        if self.using_time_embedding and times is not None:
            from .swinUNET import time_to_features_batch
            time_feats = time_to_features_batch(times, H, W, x.device)
            x = torch.cat([x, time_feats], dim=1)

        # 先 patch embedding
        if self.using_checkpoints:
            x_patch = checkpoint.checkpoint(self.patch_emb, x, use_reentrant=False)
        else:
            x_patch = self.patch_emb(x)


        latent_x, _ = self.mid_layer.encoder(x_patch)

        if y is not None:
            if self.using_time_embedding and times is not None:
                y = torch.cat([y, time_feats], dim=1)
            if self.using_checkpoints:
                y_patch = checkpoint.checkpoint(self.patch_emb, y, use_reentrant=False)
            else:
                y_patch = self.patch_emb(y)

            latent_y, _ = self.mid_layer.encoder(y_patch)

        
        mmd_loss = self.cal_mmd(latent_x, latent_y) if y is not None else torch.tensor(0.0, device=x.device)

        # 主任务输出
        if self.using_kl:
            mid_out, mu, log_var = self.mid_layer(x_patch)
            out = self.patch_head(mid_out)
        else:
            mid_out = self.mid_layer(x_patch)
            out = self.patch_head(mid_out)
        out = F.interpolate(out, size=self.img_size, mode='bilinear', align_corners=False)


        return out, latent_x, latent_y, mmd_loss


    
           
