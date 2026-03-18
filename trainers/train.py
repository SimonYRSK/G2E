import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.distributed as dist
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from PIL import Image

# 兼容新版 Pillow 移除 Image.ANTIALIAS 的情况
if not hasattr(Image, "ANTIALIAS"):
    if hasattr(Image, "Resampling"):
        Image.ANTIALIAS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    else:
        Image.ANTIALIAS = Image.LANCZOS  # type: ignore[attr-defined]


class BaseTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, epochs, device, beta,
                 tb_dir: str = "./tensorboard_logs",
                 save_dir: str = "./checkpoints",
                 save_interval: int = 1,
                 use_amp: bool = False,
                 kl_anneal: bool = False,
                 kl_anneal_epochs: int = 10):
        self.model = model
        self.trainlo = train_loader
        self.vallo = val_loader
        self.opt = optimizer
        self.sch = scheduler
        self.epochs = epochs
        self.start_epoch = 0
        self.device = device
        # KL 权重（beta）：默认固定，也可配合 KL annealing 动态调节
        self.beta = beta
        self.beta_target = beta
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.best_loss = float('inf')

        # KL annealing 配置（仅当上层 Trainer/模型实际使用 KL 时才会生效）
        self.kl_anneal = kl_anneal
        self.kl_anneal_epochs = max(1, int(kl_anneal_epochs))

        # 记录每个 epoch 的 train/val loss，用于绘图
        self.train_loss_history = []
        self.val_loss_history = []

        self.writer = SummaryWriter(
            log_dir=tb_dir
        )
        print(f"TensorBoard logs will be saved to: {self.writer.log_dir}")

        os.makedirs(self.save_dir, exist_ok=True)


    def _plot_and_log_loss_curves(self, epoch: int):
        """绘制 train / val loss 曲线，保存到文件并写入 TensorBoard。

        每个 epoch 调用一次，覆盖同名图片文件。
        在分布式/FSDP 场景下，如存在 self.is_master，则只在 rank==0 执行。
        """

        # 分布式场景下，只在主进程绘图/写文件
        if hasattr(self, "is_master") and not getattr(self, "is_master"):
            return

        if len(self.train_loss_history) == 0:
            return

        epochs = list(range(1, len(self.train_loss_history) + 1))

        plt.switch_backend("Agg")
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(epochs, self.train_loss_history, label="train_loss", color="tab:blue")
        if len(self.val_loss_history) == len(self.train_loss_history):
            ax.plot(epochs, self.val_loss_history, label="val_loss", color="tab:orange")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train / Val Loss")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()

        # 覆盖保存到同一个文件
        img_path = os.path.join(self.save_dir, "loss_curve.png")
        fig.savefig(img_path)

        # 写入 TensorBoard（捕获可能的图像写入异常，避免训练中断）
        if hasattr(self, "writer") and self.writer is not None:
            try:
                self.writer.add_figure("Loss/curve", fig, global_step=epoch)
            except Exception as e:
                print(f"[Warning] Failed to write loss figure to TensorBoard: {e}")

        plt.close(fig)


    def lat_weight(self, shape):
        """纬度加权"""
        H = shape[-2]
        lat = torch.linspace(-90 + 180/(2*H), 90 - 180/(2*H), H, device=self.device)
        weight = torch.cos(torch.deg2rad(lat))
        weight = weight / weight.mean()

        view_shape = [1] * len(shape)
        view_shape[-2] = H

        return weight.view(*view_shape)

    def cal_losses(self, x_recon, y, mu, log_var, weight=None):        
        
        device_type = self.device.type if isinstance(self.device, torch.device) else "cuda"
        
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x_recon = x_recon.float()
            y = y.float()
            mu = mu.float()
            log_var = log_var.float()
            
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

            se = (x_recon - y) ** 2

            if weight is not None:
                se = se * weight.float()

            recon_loss = torch.mean(se)

            return kl_loss, recon_loss
    

    def save_checkpoint(self, epoch, current_avg_loss):
        improve = current_avg_loss < self.best_loss
        if improve:
            self.best_loss = current_avg_loss
            file_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch + 1 + self.start_epoch}.pth")
            state = {
                'epoch': epoch + 1 + self.start_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.opt.state_dict(),
                'scheduler_state_dict': self.sch.state_dict() if self.sch else None,
                'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            }
            torch.save(state, file_path)
            print(f"Checkpoint saved to {file_path}")

            self.writer.add_scalar("best/val_loss", current_avg_loss,  epoch + self.start_epoch)

    def load_checkpoint(self, path, strict=True, only_model=False):
        
        print(f"Loading checkpoint from {path} ...")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        if not only_model:
        
            if 'optimizer_state_dict' in checkpoint and self.opt:
                self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
                
            if 'scheduler_state_dict' in checkpoint and self.sch and checkpoint['scheduler_state_dict']:
                self.sch.load_state_dict(checkpoint['scheduler_state_dict'])

            if 'scaler_state_dict' in checkpoint and self.scaler:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.start_epoch = checkpoint.get('epoch', 0)  # 记录断点
        return self.start_epoch, None

    def validate_one_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(':')[0]

        with torch.no_grad():
            for x, y, i, times in self.vallo:
                x = x.to(self.device)
                y = y.to(self.device)
                i = i.to(self.device)
                # times = times.to(self.device)
                weights = self.lat_weight(y.shape)
                with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                    x_recon, mu, log_var = self.model(x, i=i, times=times)
                    kl_loss, recon_loss = self.cal_losses(x_recon, y, mu, log_var, weight=weights)
                    loss = kl_loss * self.beta + recon_loss

                total_loss += float(loss.detach())
                total_recon_loss += float(recon_loss.detach())
                total_kl_loss += float(kl_loss.detach())
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_recon = total_recon_loss / num_batches
        avg_kl = total_kl_loss / num_batches

        print(f"\nEpoch {epoch+1} 验证集平均:")
        print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}, 散度={avg_kl:.5f}")

        global_step = epoch

        if hasattr(self, 'writer') and self.writer:
            self.writer.add_scalar("Loss/val/total",    avg_loss,  global_step)
            self.writer.add_scalar("Loss/val/recon",    avg_recon, global_step)
            self.writer.add_scalar("Loss/val/kl",       avg_kl,    global_step)

        return avg_loss

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        pbar = tqdm(self.trainlo, desc=f"Epoch {epoch+1}/{self.epochs}")

        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(':')[0]

        for batch_idx, (x, y, i, times) in enumerate(pbar):
            x = x.to(self.device)
            y = y.to(self.device)
            i = i.to(self.device)
            # times = times.to(self.device)
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Batch {batch_idx} input contains nan/inf!")
            if torch.isnan(y).any() or torch.isinf(y).any():
                print(f"Batch {batch_idx} target contains nan/inf!")

            weights = self.lat_weight(y.shape)

            self.opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                x_recon, mu, log_var = self.model(x, i=i, times=times)
                kl_loss, recon_loss = self.cal_losses(x_recon, y, mu, log_var, weight=weights)
                loss = kl_loss * self.beta + recon_loss

            loss_item = float(loss.detach())
            recon_item = float(recon_loss.detach())
            kl_item = float(kl_loss.detach())

            total_loss += loss_item
            total_recon_loss += recon_item
            total_kl_loss += kl_item

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.scaler.step(self.opt)
            self.scaler.update()

            pbar.set_postfix({
                'Loss': f'{loss_item:.4f}',
                'Recon': f'{recon_item:.4f}',
                'KL': f'{kl_item:.4f}',
            })

            if batch_idx % 10 == 0 and hasattr(self, 'writer') and self.writer:
                step = epoch * len(self.trainlo) + batch_idx
                self.writer.add_scalar("Loss/batch/total", loss_item, step)
                self.writer.add_scalar("Loss/batch/recon", recon_item, step)
                self.writer.add_scalar("Loss/batch/kl",    kl_item,    step)


                
        val_loss = self.validate_one_epoch(epoch)

        if isinstance(self.sch, ReduceLROnPlateau):
            self.sch.step(val_loss)
        else:
            self.sch.step()

        avg_loss = total_loss / len(self.trainlo)
        avg_recon = total_recon_loss / len(self.trainlo)
        avg_kl = total_kl_loss / len(self.trainlo)
        
        print(f"\nEpoch {epoch+1} 训练集平均:")
        print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}, 散度={avg_kl:.5f}")

        global_step = epoch

        self.writer.add_scalar("Loss/train/total",    avg_loss,  global_step)
        self.writer.add_scalar("Loss/train/recon",    avg_recon, global_step)
        self.writer.add_scalar("Loss/train/kl",       avg_kl,    global_step)
        self.writer.add_scalar("hyper/lr",            self.opt.param_groups[0]['lr'], global_step)


        return avg_loss, val_loss


    def train(self, resume_path=None, only_model = False):
        start_epoch = 0
        best_metric = None
        
        if resume_path is not None:
            start_epoch, best_metric = self.load_checkpoint(resume_path, strict=False, only_model = only_model)

        try:

            for epoch in range(0, self.epochs):
                # 在每个 epoch 开始前根据 KL annealing 调整当前 beta
                if self.kl_anneal and self.beta_target > 0.0:
                    # 仅当上层 Trainer/模型真的在用 KL 时才有意义
                    if getattr(self, "using_kl", True):
                        if epoch < self.kl_anneal_epochs:
                            self.beta = self.beta_target * float(epoch + 1) / float(self.kl_anneal_epochs)
                        else:
                            self.beta = self.beta_target

                avg_loss, val_loss = self.train_one_epoch(epoch)

                # 记录损失历史并绘制曲线
                self.train_loss_history.append(avg_loss)
                self.val_loss_history.append(val_loss)
                self._plot_and_log_loss_curves(epoch)
                
                if (epoch + 1) % self.save_interval == 0:
                    self.save_checkpoint(epoch, val_loss)

        finally:
            if hasattr(self, 'writer') and self.writer:
                self.writer.close()        
                print("TensorBoard writer closed.")


    

