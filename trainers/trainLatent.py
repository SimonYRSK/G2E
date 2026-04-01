import os

import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import ReduceLROnPlateau

from trainers.fsdptrain import FSDPUNetTrainer


class FSDPLatentTrainer(FSDPUNetTrainer):
    """UNetTrainer 的 FSDP 版本。

    - 支持单机多卡 FSDP 训练
    - 只在 rank==0 时写日志 / 保存模型 / 打印 epoch 级信息
    - 对 DistributedSampler 调用 set_epoch
    """

    def __init__(self, *args, mmd_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.mmd_weight = mmd_weight


    def _all_reduce_loss(self, total_loss: float, total_recon: float, total_mmd: float, num_batches: int):
        """在所有进程间做 all_reduce，得到全局平均 loss。"""
        if not dist.is_available() or not dist.is_initialized():
            avg_loss = total_loss / max(num_batches, 1)
            avg_recon = total_recon / max(num_batches, 1)
            avg_mmd = total_mmd / max(num_batches, 1)
            return avg_loss, avg_recon, avg_mmd

        device = self.device if isinstance(self.device, torch.device) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor([total_loss, total_recon, total_mmd, float(num_batches)], device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_loss_g, total_recon_g, total_mmd_g, num_batches_g = tensor.tolist()
        num_batches_g = max(num_batches_g, 1.0)
        avg_loss = float(total_loss_g / num_batches_g)
        avg_recon = float(total_recon_g / num_batches_g)
        avg_mmd = float(total_mmd_g / num_batches_g)
        return avg_loss, avg_recon, avg_mmd


    def validate_one_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_mmd_loss = 0.0
        num_batches = 0

        # 只在第一个正常 batch 上画图
        has_plotted = False

        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(":")[0]

        with torch.no_grad():
            for batch_idx, (x, y, i, times) in enumerate(self.vallo):
                x = x.to(self.device)
                y = y.to(self.device)

                # 检查验证集 batch 是否存在 NaN/Inf，并打印对应时间
                has_nan_inf_x = torch.isnan(x).any() or torch.isinf(x).any()
                has_nan_inf_y = torch.isnan(y).any() or torch.isinf(y).any()
                if has_nan_inf_x or has_nan_inf_y:
                    if self.is_master:
                        times_str = ", ".join(str(t) for t in list(times))
                        print(f"[Val] batch {batch_idx} contains NaN/Inf, times: {times_str}")
                        print("[Val] 该 batch 已跳过，用于避免验证损失变为 NaN")
                    continue

                # i: lead time index，这里不再参与模型计算
                weights = self.lat_weight(y.shape)
                with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):

                    x_recon, _, _, mmd_loss = self.model(x, y, times=times)

                    recon_loss = self.cal_losses(x_recon, y, weight=weights)

                    
                    loss = recon_loss + self.mmd_weight * mmd_loss

                total_loss += float(loss.detach())
                total_recon_loss += float(recon_loss.detach())
                total_mmd_loss += float(mmd_loss.detach())
                num_batches += 1

                # 在首个正常 batch 上画图（只在主进程）
                if self.is_master and not has_plotted:
                    try:
                        self._plot_validation_maps(epoch, x_recon, y, times)
                    except Exception as e:
                        # 避免画图错误中断训练，仅在主进程打印
                        if self.is_master:
                            print(f"[Val] 绘图时出错: {e}")
                    has_plotted = True

        # 若全部 batch 都被跳过，避免除以 0
        if num_batches == 0:
            if self.is_master:
                print("[Val] 所有 batch 均因包含 NaN/Inf 被跳过，返回损失 0.0 以保持训练继续进行")
            avg_loss = 0.0
            avg_recon = 0.0
            avg_mmd = 0.0
        else:
            avg_loss, avg_recon, avg_mmd = self._all_reduce_loss(total_loss, total_recon_loss, total_mmd_loss, num_batches)

        if self.is_master:
            print(f"\nEpoch {epoch+1} 验证集平均:")
            # 这里暂时只打印总损失和重建损失，KL 如需可在上面累加并打印
            print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}, MMD={avg_mmd:.5f}")

            global_step = epoch
            if hasattr(self, "writer") and self.writer:
                self.writer.add_scalar("Loss/val/total", avg_loss, global_step)
                self.writer.add_scalar("Loss/val/recon", avg_recon, global_step)
                self.writer.add_scalar("Loss/val/mmd", avg_mmd, global_step)

        return avg_loss


    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_mmd_loss = 0.0
        num_batches = 0

        # DistributedSampler 设 epoch，保证每轮 shuffle 不同
        sampler = getattr(self.trainlo, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        from tqdm import tqdm

        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(":")[0]

        pbar = tqdm(self.trainlo, desc=f"Epoch {epoch+1}/{self.epochs}", disable=not self.is_master)

        for batch_idx, (x, y, i, times) in enumerate(pbar):
            x = x.to(self.device)
            y = y.to(self.device)
            # i: lead time index，这里不再参与模型计算
            has_nan_inf_x = torch.isnan(x).any() or torch.isinf(x).any()
            has_nan_inf_y = torch.isnan(y).any() or torch.isinf(y).any()
            if has_nan_inf_x or has_nan_inf_y:
                if self.is_master:
                    times_str = ", ".join(str(t) for t in list(times))
                    print(f"[Train] batch {batch_idx} contains NaN/Inf, times: {times_str}")
                    print("[Train] 该 batch 已跳过，用于避免训练权重被 NaN 污染")
                continue

            weights = self.lat_weight(y.shape)

            self.opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                x_recon, _, _, mmd_loss = self.model(x, y, times=times)

                recon_loss = self.cal_losses(x_recon, y, weight=weights)

                
                loss = recon_loss + self.mmd_weight * mmd_loss

            # 如果 loss 本身出现 NaN/Inf，同样跳过该 batch，避免反向传播污染参数
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                if self.is_master:
                    times_str = ", ".join(str(t) for t in list(times))
                    print(f"[Train] batch {batch_idx} loss is NaN/Inf, times: {times_str}")
                    print("[Train] 该 batch 的梯度已跳过，请检查数据或数值稳定性")
                continue

            loss_item = float(loss.detach())
            recon_item = float(recon_loss.detach())
            mmd_loss_item = float(mmd_loss.detach())

            total_loss += loss_item
            total_recon_loss += recon_item
            total_mmd_loss += mmd_loss_item
            num_batches += 1

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)

            # FSDP 建议使用专门的梯度裁剪：传入 FSDP 模型本身
            if isinstance(self.model, FSDP):
                FSDP.clip_grad_norm_(self.model, max_norm=5.0)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.scaler.step(self.opt)
            self.scaler.update()

            if self.is_master:

                pbar.set_postfix({
                    'Loss': f'{loss_item:.4f}',
                    'Recon': f'{recon_item:.4f}',
                    'MMD': f'{mmd_loss_item:.4f}',
                })

                if batch_idx % 10 == 0 and hasattr(self, 'writer') and self.writer:
                    step = epoch * len(self.trainlo) + batch_idx
                    self.writer.add_scalar("Loss/batch/total", loss_item, step)
                    self.writer.add_scalar("Loss/batch/recon", recon_item, step)
                    self.writer.add_scalar("Loss/batch/mmd",   mmd_loss_item, step)

        # 得到全局平均 train loss（这里只 all_reduce 总损失和重建损失，KL 已体现在总损失中）
        if num_batches == 0:
            if self.is_master:
                print("[Train] 本 epoch 所有 batch 均因 NaN/Inf 被跳过，返回损失 0.0")
            avg_loss = 0.0
            avg_recon = 0.0
            avg_mmd = 0.0
        else:
            avg_loss, avg_recon, avg_mmd = self._all_reduce_loss(total_loss, total_recon_loss, total_mmd_loss, num_batches)

        # 验证也返回全局损失
        val_loss = self.validate_one_epoch(epoch)

        # scheduler 所有 rank 都要 step，保证 lr 一致
        if isinstance(self.sch, ReduceLROnPlateau):
            self.sch.step(val_loss)
        else:
            self.sch.step()

        if self.is_master:
            print(f"\nEpoch {epoch+1} 训练集平均:")
            print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}, MMD={avg_mmd:.5f}")

            global_step = epoch
            if hasattr(self, 'writer') and self.writer:
                self.writer.add_scalar("Loss/train/total",    avg_loss,  global_step)
                self.writer.add_scalar("Loss/train/recon",    avg_recon, global_step)
                self.writer.add_scalar("Loss/train/mmd",      avg_mmd,   global_step)
                self.writer.add_scalar("hyper/lr",            self.opt.param_groups[0]['lr'], global_step)

        return avg_loss, val_loss
