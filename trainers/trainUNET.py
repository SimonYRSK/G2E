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
    """

    def cal_losses(self, x_recon, y, weight=None):
        """计算加权 MSE 重建损失。"""
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
        num_batches = 0

        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(':')[0]

        with torch.no_grad():
            for x, y, i, times in self.vallo:
                x = x.to(self.device)
                y = y.to(self.device)
                # i: lead time index，这里不再参与模型计算
                weights = self.lat_weight(y.shape)
                with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                    x_recon = self.model(x, times=times)
                    recon_loss = self.cal_losses(x_recon, y, weight=weights)
                    loss = recon_loss

                total_loss += float(loss.detach())
                total_recon_loss += float(recon_loss.detach())
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_recon = total_recon_loss / num_batches

        print(f"\nEpoch {epoch+1} 验证集平均:")
        print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}")

        global_step = epoch

        if hasattr(self, 'writer') and self.writer:
            self.writer.add_scalar("Loss/val/total",    avg_loss,  global_step)
            self.writer.add_scalar("Loss/val/recon",    avg_recon, global_step)

        return avg_loss

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0

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
                x_recon = self.model(x, times=times)
                recon_loss = self.cal_losses(x_recon, y, weight=weights)
                loss = recon_loss

            loss_item = float(loss.detach())
            recon_item = float(recon_loss.detach())

            total_loss += loss_item
            total_recon_loss += recon_item

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.scaler.step(self.opt)
            self.scaler.update()

            pbar.set_postfix({
                'Loss': f'{loss_item:.4f}',
                'Recon': f'{recon_item:.4f}',
            })

            if batch_idx % 10 == 0 and hasattr(self, 'writer') and self.writer:
                step = epoch * len(self.trainlo) + batch_idx
                self.writer.add_scalar("Loss/batch/total", loss_item, step)
                self.writer.add_scalar("Loss/batch/recon", recon_item, step)

        val_loss = self.validate_one_epoch(epoch)

        if isinstance(self.sch, ReduceLROnPlateau):
            self.sch.step(val_loss)
        else:
            self.sch.step()

        avg_loss = total_loss / len(self.trainlo)
        avg_recon = total_recon_loss / len(self.trainlo)

        print(f"\nEpoch {epoch+1} 训练集平均:")
        print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}")

        global_step = epoch

        self.writer.add_scalar("Loss/train/total",    avg_loss,  global_step)
        self.writer.add_scalar("Loss/train/recon",    avg_recon, global_step)
        self.writer.add_scalar("hyper/lr",            self.opt.param_groups[0]['lr'], global_step)

        return avg_loss, val_loss


    

