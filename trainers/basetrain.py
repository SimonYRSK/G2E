from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

class BaseTrainer:
    def __init__(self, model, train_loader, test_loader, optimizer, scheduler, epochs, device, beta, 
                 save_dir: str = "./checkpoints", save_interval: int = 1):
        self.model = model
        self.trainlo = train_loader
        self.testlo = test_loader
        self.opt = optimizer
        self.sch = scheduler
        self.epochs = epochs
        self.device = device
        self.beta = beta
        
        # 新增：checkpoint 参数
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval

    def cal_losses(self, x, y, mu, logvar):
        
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = torch.mean((x - y) ** 2)

        return kl_loss, recon_loss

    def save_checkpoint(self, epoch):
        """保存模型 checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'scheduler_state_dict': self.sch.state_dict(),
        }
        
        save_path = self.save_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, save_path)
        print(f"✅ Checkpoint 已保存: {save_path}")

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        pbar = tqdm(self.trainlo, desc=f"Epoch {epoch+1}/{self.epochs}")

        for batch_idx, (x, y, t) in enumerate(pbar):
            x = x.unsqueeze(2).to(self.device)
            y = y.to(self.device)
            #t = t.to(self.device)

            x_recon, mu, log_var = self.model(x)
            kl_loss, recon_loss = self.cal_losses(x_recon, y, mu, log_var)
            loss = kl_loss * self.beta + recon_loss

            total_loss += loss
            total_recon_loss += recon_loss 
            total_kl_loss += kl_loss

            self.opt.zero_grad()
            loss.backward()

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'KL': f'{kl_loss.item():.4f}',
            })

            self.opt.step()
        
        self.sch.step()

        avg_loss = total_loss / len(self.trainlo)
        avg_recon = total_recon_loss / len(self.trainlo)
        avg_kl = total_kl_loss / len(self.trainlo)
        print(f"\nEpoch {epoch+1} 训练集平均:")
        print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}, 散度={avg_kl:.5f}")   

    def train(self):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            
            # 新增：按间隔保存 checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch)