import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchDiscriminator(nn.Module):
    """条件 PatchGAN 判别器：输入 [x_gfs, y]，通道数为 in_x_chans + in_y_chans。"""

    def __init__(self, in_x_chans: int = 70, in_y_chans: int = 70, base_channels: int = 64):
        super().__init__()
        in_chans = in_x_chans + in_y_chans

        self.net = nn.Sequential(
            nn.Conv2d(in_chans, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x_gfs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_gfs, y], dim=1)
        return self.net(inp)


def d_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def g_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def logits_feature_matching_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """按需求：使用判别器 logits 的 L1 作为 feature matching 项。"""
    return F.l1_loss(fake_logits, real_logits.detach())
