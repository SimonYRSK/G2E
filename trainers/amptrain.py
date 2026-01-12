from .basetrain import BaseTrainer
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

class AMPTrainer(BaseTrainer):
    def __init__(self, model, train_loader, test_loader, optimizer, scheduler, epochs, device, beta):
        super().__init__(model, train_loader, test_loader, optimizer, scheduler, epochs, device, beta)
        # 只有 cuda 时启用 amp
        self.scaler = GradScaler(enabled=str(device).startswith("cuda"))

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        pbar = tqdm(self.trainlo, desc=f"Epoch {epoch+1}/{self.epochs}")

        for batch_idx, (x, y, t) in enumerate(pbar):
            # 你的数据: (N, C, H, W) → 加 T=1 维
            x = x.unsqueeze(2).to(self.device)
            y = y.to(self.device)

            self.opt.zero_grad(set_to_none=True)

            with autocast(enabled=str(self.device).startswith("cuda")):
                x_recon, mu, log_var = self.model(x)
                kl_loss, recon_loss = self.cal_losses(x_recon, y, mu, log_var)
                loss = kl_loss * self.beta + recon_loss

            # 记录标量
            loss_item = float(loss.detach())
            recon_item = float(recon_loss.detach())
            kl_item = float(kl_loss.detach())
            total_loss += loss_item
            total_recon_loss += recon_item
            total_kl_loss += kl_item

            # AMP 反传与更新
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()

            pbar.set_postfix({
                'Loss': f'{loss_item:.4f}',
                'Recon': f'{recon_item:.4f}',
                'KL': f'{kl_item:.4f}',
            })

        self.sch.step()

        avg_loss = total_loss / len(self.trainlo)
        avg_recon = total_recon_loss / len(self.trainlo)
        avg_kl = total_kl_loss / len(self.trainlo)
        print(f"\nEpoch {epoch+1} 训练集平均:")
        print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}, 散度={avg_kl:.5f}")