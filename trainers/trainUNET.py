import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from trainers.train import BaseTrainer


class UNetTrainer(BaseTrainer):
    """针对 swinUNET 的 Trainer：

    - 继承通用 BaseTrainer（在 trainers/train.py 中）
    - 模型输出为单个重建张量 x_recon
    - 损失仅为加权 MSE（无 KL 项）
    - 前向调用为 model(x, times=times)
    - 当模型具有 using_kl=True 时，假定前向返回 (x_recon, mu, log_var)，并在损失中加入 KL 项
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs,
        device,
        beta,
        tb_dir: str = "./tensorboard_logs",
        save_dir: str = "./checkpoints",
        save_interval: int = 1,
        use_amp: bool = False,
        kl_anneal: bool = False,
        kl_anneal_epochs: int = 10,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            device=device,
            beta=beta,
            tb_dir=tb_dir,
            save_dir=save_dir,
            save_interval=save_interval,
            use_amp=use_amp,
            kl_anneal=kl_anneal,
            kl_anneal_epochs=kl_anneal_epochs,
        )

        # 是否在瓶颈使用 KL（由模型控制）
        self.using_kl = bool(getattr(self.model, "using_kl", False))

    def cal_losses(self, x_recon, y, weight=None):
        """计算加权 MSE 重建损失（不含 KL）。"""
        device_type = self.device.type if isinstance(self.device, torch.device) else "cuda"

        with torch.amp.autocast(device_type=device_type, enabled=False):
            x_recon = x_recon.float()
            y = y.float()

            se = (x_recon - y) ** 2
            if weight is not None:
                se = se * weight.float()

            recon_loss = torch.mean(se)

        return recon_loss

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
                # i: lead time index，这里不再参与模型计算
                weights = self.lat_weight(y.shape)
                with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                    if self.using_kl:
                        x_recon, mu, log_var = self.model(x, times=times)
                    else:
                        x_recon = self.model(x, times=times)
                        mu = log_var = None

                    recon_loss = self.cal_losses(x_recon, y, weight=weights)

                    if self.using_kl and mu is not None and log_var is not None:
                        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                        loss = recon_loss + self.beta * kl_loss
                    else:
                        kl_loss = torch.tensor(0.0, device=self.device)
                        loss = recon_loss

                total_loss += float(loss.detach())
                total_recon_loss += float(recon_loss.detach())
                total_kl_loss += float(kl_loss.detach())
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_recon = total_recon_loss / num_batches
        avg_kl = total_kl_loss / num_batches if num_batches > 0 else 0.0

        print(f"\nEpoch {epoch+1} 验证集平均:")
        if self.using_kl:
            print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}, KL={avg_kl:.5f}")
        else:
            print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}")

        global_step = epoch

        if hasattr(self, 'writer') and self.writer:
            self.writer.add_scalar("Loss/val/total",    avg_loss,  global_step)
            self.writer.add_scalar("Loss/val/recon",    avg_recon, global_step)
            if self.using_kl:
                self.writer.add_scalar("Loss/val/kl",   avg_kl,    global_step)

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
            # i: lead time index，这里不再参与模型计算
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Batch {batch_idx} input contains nan/inf!")
            if torch.isnan(y).any() or torch.isinf(y).any():
                print(f"Batch {batch_idx} target contains nan/inf!")

            weights = self.lat_weight(y.shape)

            self.opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                if self.using_kl:
                    x_recon, mu, log_var = self.model(x, times=times)
                else:
                    x_recon = self.model(x, times=times)
                    mu = log_var = None

                recon_loss = self.cal_losses(x_recon, y, weight=weights)

                if self.using_kl and mu is not None and log_var is not None:
                    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                    loss = recon_loss + self.beta * kl_loss
                else:
                    kl_loss = torch.tensor(0.0, device=self.device)
                    loss = recon_loss

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

            if self.using_kl:
                pbar.set_postfix({
                    'Loss': f'{loss_item:.4f}',
                    'Recon': f'{recon_item:.4f}',
                    'KL': f'{kl_item:.4f}',
                })
            else:
                pbar.set_postfix({
                    'Loss': f'{loss_item:.4f}',
                    'Recon': f'{recon_item:.4f}',
                })

            if batch_idx % 10 == 0 and hasattr(self, 'writer') and self.writer:
                step = epoch * len(self.trainlo) + batch_idx
                self.writer.add_scalar("Loss/batch/total", loss_item, step)
                self.writer.add_scalar("Loss/batch/recon", recon_item, step)
                if self.using_kl:
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
        if self.using_kl:
            print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}, KL={avg_kl:.5f}")
        else:
            print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}")

        global_step = epoch

        self.writer.add_scalar("Loss/train/total",    avg_loss,  global_step)
        self.writer.add_scalar("Loss/train/recon",    avg_recon, global_step)
        if self.using_kl:
            self.writer.add_scalar("Loss/train/kl",   avg_kl,    global_step)
            if self.kl_anneal:
                self.writer.add_scalar("hyper/beta",  self.beta, global_step)
        self.writer.add_scalar("hyper/lr",            self.opt.param_groups[0]['lr'], global_step)

        return avg_loss, val_loss


    

