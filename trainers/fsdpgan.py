import os

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP

from trainers.fsdptrain import FSDPUNetTrainer
from models.gan import d_hinge_loss, g_hinge_loss, logits_feature_matching_loss


class FSDPGANTrainer(FSDPUNetTrainer):
    """在 FSDPUNetTrainer 基础上增加两阶段训练：

    - epoch < gan_start_epoch: 纯 baseline（与 mainfsdp 一致）
    - epoch >= gan_start_epoch: L1 + GAN(hinge) + FM(logits-L1)
    """

    def __init__(
        self,
        *args,
        discriminator,
        d_optimizer=None,
        gan_start_epoch: int = 35,
        l1_weight: float = 10.0,
        adv_weight: float = 1.0,
        fm_weight: float = 5.0,
        d_grad_clip: float = 5.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.gan_start_epoch = int(gan_start_epoch)
        self.l1_weight = float(l1_weight)
        self.adv_weight = float(adv_weight)
        self.fm_weight = float(fm_weight)
        self.d_grad_clip = float(d_grad_clip)

        self.discriminator = discriminator.to(self.device)
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            if isinstance(self.device, torch.device) and self.device.type == "cuda":
                self.discriminator = DDP(self.discriminator, device_ids=[self.device.index])
            else:
                self.discriminator = DDP(self.discriminator)

        self.d_optimizer = d_optimizer
        if self.d_optimizer is None:
            self.d_optimizer = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=2e-4,
                betas=(0.5, 0.999),
            )

        self.d_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        if self.is_master:
            print(
                f"[FSDPGANTrainer] gan_start_epoch={self.gan_start_epoch}, "
                f"l1={self.l1_weight}, adv={self.adv_weight}, fm={self.fm_weight}"
            )

    def save_checkpoint(self, epoch, current_avg_loss):
        if not self.is_master:
            return

        improve = current_avg_loss < self.best_loss
        if improve:
            self.best_loss = current_avg_loss
            file_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch + 1 + self.start_epoch}.pth")
            state = {
                "epoch": epoch + 1 + self.start_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                "scheduler_state_dict": self.sch.state_dict() if self.sch else None,
                "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
                "discriminator_state_dict": self.discriminator.state_dict(),
                "d_optimizer_state_dict": self.d_optimizer.state_dict(),
                "d_scaler_state_dict": self.d_scaler.state_dict() if self.use_amp else None,
            }
            torch.save(state, file_path)
            print(f"Checkpoint saved to {file_path}")

            if hasattr(self, "writer") and self.writer:
                self.writer.add_scalar("best/val_loss", current_avg_loss, epoch + self.start_epoch)

    def train_one_epoch(self, epoch):
        self.model.train()
        self.discriminator.train()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_d_loss = 0.0
        total_adv_loss = 0.0
        total_fm_loss = 0.0
        num_batches = 0

        sampler = getattr(self.trainlo, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        from tqdm import tqdm

        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(":")[0]
        pbar = tqdm(self.trainlo, desc=f"Epoch {epoch+1}/{self.epochs}", disable=not self.is_master)

        use_gan = epoch >= self.gan_start_epoch

        for batch_idx, (x, y, i, times) in enumerate(pbar):
            x = x.to(self.device)
            y = y.to(self.device)

            has_nan_inf_x = torch.isnan(x).any() or torch.isinf(x).any()
            has_nan_inf_y = torch.isnan(y).any() or torch.isinf(y).any()
            if has_nan_inf_x or has_nan_inf_y:
                if self.is_master:
                    times_str = ", ".join(str(t) for t in list(times))
                    print(f"[Train] batch {batch_idx} contains NaN/Inf, times: {times_str}")
                    print("[Train] 该 batch 已跳过")
                continue

            self.opt.zero_grad(set_to_none=True)

            if not use_gan:
                # baseline 阶段：沿用原训练逻辑（Weighted MSE / + KL）
                weights = self.lat_weight(y.shape)
                with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                    if getattr(self, "using_kl", False):
                        x_recon, mu, log_var = self.model(x, times=times)
                    else:
                        x_recon = self.model(x, times=times)
                        mu = log_var = None

                    recon_loss = self.cal_losses(x_recon, y, weight=weights)

                    if getattr(self, "using_kl", False) and mu is not None and log_var is not None:
                        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                        g_loss = recon_loss + self.beta * kl_loss
                    else:
                        g_loss = recon_loss

                if torch.isnan(g_loss).any() or torch.isinf(g_loss).any():
                    if self.is_master:
                        print(f"[Train] batch {batch_idx} baseline loss is NaN/Inf, skipped")
                    continue

                self.scaler.scale(g_loss).backward()
                self.scaler.unscale_(self.opt)
                if isinstance(self.model, FSDP):
                    FSDP.clip_grad_norm_(self.model, max_norm=5.0)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.opt)
                self.scaler.update()

                d_loss_item = 0.0
                adv_item = 0.0
                fm_item = 0.0
                recon_item = float(recon_loss.detach())
                loss_item = float(g_loss.detach())

            else:
                # -------------------------
                # GAN 阶段：先 D 后 G
                # -------------------------

                # 1) 更新 D
                self.d_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                        y_fake_det = self.model(x, times=times)
                        if isinstance(y_fake_det, (tuple, list)):
                            y_fake_det = y_fake_det[0]

                with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                    real_logits = self.discriminator(x, y)
                    fake_logits = self.discriminator(x, y_fake_det.detach())
                    d_loss = d_hinge_loss(real_logits, fake_logits)

                if torch.isnan(d_loss).any() or torch.isinf(d_loss).any():
                    if self.is_master:
                        print(f"[Train] batch {batch_idx} d_loss is NaN/Inf, skipped")
                    continue

                self.d_scaler.scale(d_loss).backward()
                self.d_scaler.unscale_(self.d_optimizer)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.d_grad_clip)
                self.d_scaler.step(self.d_optimizer)
                self.d_scaler.update()

                # 2) 更新 G
                for p in self.discriminator.parameters():
                    p.requires_grad_(False)

                with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                    y_fake = self.model(x, times=times)
                    if isinstance(y_fake, (tuple, list)):
                        y_fake = y_fake[0]

                    fake_logits_g = self.discriminator(x, y_fake)
                    with torch.no_grad():
                        real_logits_g = self.discriminator(x, y)

                    l1_loss = F.l1_loss(y_fake, y)
                    adv_loss = g_hinge_loss(fake_logits_g)
                    fm_loss = logits_feature_matching_loss(real_logits_g, fake_logits_g)
                    g_loss = self.l1_weight * l1_loss + self.adv_weight * adv_loss + self.fm_weight * fm_loss

                for p in self.discriminator.parameters():
                    p.requires_grad_(True)

                if torch.isnan(g_loss).any() or torch.isinf(g_loss).any():
                    if self.is_master:
                        print(f"[Train] batch {batch_idx} g_loss is NaN/Inf, skipped")
                    continue

                self.scaler.scale(g_loss).backward()
                self.scaler.unscale_(self.opt)
                if isinstance(self.model, FSDP):
                    FSDP.clip_grad_norm_(self.model, max_norm=5.0)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.opt)
                self.scaler.update()

                d_loss_item = float(d_loss.detach())
                adv_item = float(adv_loss.detach())
                fm_item = float(fm_loss.detach())
                recon_item = float(l1_loss.detach())
                loss_item = float(g_loss.detach())

            total_loss += loss_item
            total_recon_loss += recon_item
            total_d_loss += d_loss_item
            total_adv_loss += adv_item
            total_fm_loss += fm_item
            num_batches += 1

            if self.is_master:
                if use_gan:
                    pbar.set_postfix(
                        {
                            "G": f"{loss_item:.4f}",
                            "L1": f"{recon_item:.4f}",
                            "D": f"{d_loss_item:.4f}",
                            "ADV": f"{adv_item:.4f}",
                            "FM": f"{fm_item:.4f}",
                        }
                    )
                else:
                    pbar.set_postfix({"Loss": f"{loss_item:.4f}", "Recon": f"{recon_item:.4f}"})

                if batch_idx % 10 == 0 and hasattr(self, "writer") and self.writer:
                    step = epoch * len(self.trainlo) + batch_idx
                    self.writer.add_scalar("Loss/batch/total", loss_item, step)
                    self.writer.add_scalar("Loss/batch/recon", recon_item, step)
                    if use_gan:
                        self.writer.add_scalar("Loss/batch/d", d_loss_item, step)
                        self.writer.add_scalar("Loss/batch/adv", adv_item, step)
                        self.writer.add_scalar("Loss/batch/fm", fm_item, step)

        if num_batches == 0:
            if self.is_master:
                print("[Train] 本 epoch 所有 batch 均被跳过，返回损失 0.0")
            avg_loss = 0.0
            avg_recon = 0.0
        else:
            avg_loss, avg_recon = self._all_reduce_loss(total_loss, total_recon_loss, num_batches)

        val_loss = self.validate_one_epoch(epoch)

        if isinstance(self.sch, ReduceLROnPlateau):
            self.sch.step(val_loss)
        else:
            self.sch.step()

        if self.is_master:
            print(f"\nEpoch {epoch+1} 训练集平均:")
            if use_gan:
                avg_d = total_d_loss / max(num_batches, 1)
                avg_adv = total_adv_loss / max(num_batches, 1)
                avg_fm = total_fm_loss / max(num_batches, 1)
                print(
                    f"G总损失={avg_loss:.5f}, L1={avg_recon:.5f}, "
                    f"D={avg_d:.5f}, ADV={avg_adv:.5f}, FM={avg_fm:.5f}"
                )
            else:
                print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}")

            if hasattr(self, "writer") and self.writer:
                self.writer.add_scalar("Loss/train/total", avg_loss, epoch)
                self.writer.add_scalar("Loss/train/recon", avg_recon, epoch)
                if use_gan:
                    self.writer.add_scalar("Loss/train/d", total_d_loss / max(num_batches, 1), epoch)
                    self.writer.add_scalar("Loss/train/adv", total_adv_loss / max(num_batches, 1), epoch)
                    self.writer.add_scalar("Loss/train/fm", total_fm_loss / max(num_batches, 1), epoch)
                self.writer.add_scalar("hyper/lr", self.opt.param_groups[0]["lr"], epoch)

        return avg_loss, val_loss
