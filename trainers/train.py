import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import torch.distributed as dist
import os

class BaseTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, epochs, device, beta, 
                 save_dir: str = "./checkpoints", save_interval: int = 1, use_amp: bool = False):
        self.model = model
        self.trainlo = train_loader
        self.vallo = val_loader
        self.opt = optimizer
        self.sch = scheduler
        self.epochs = epochs
        self.device = device
        self.beta = beta
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.best_loss = float('inf')

        self.writer = SummaryWriter(
            log_dir="/home/ximutian/tensorboard_logs/1_29"
        )
        print(f"TensorBoard logs will be saved to: {self.writer.log_dir}")

        os.makedirs(self.save_dir, exist_ok=True)


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
            file_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            state = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.opt.state_dict(),
                'scheduler_state_dict': self.sch.state_dict() if self.sch else None,
                'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            }
            torch.save(state, file_path)
            print(f"Checkpoint saved to {file_path}")

            self.writer.add_scalar("best/train_loss", current_avg_loss, epoch)

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
            
        start_epoch = checkpoint.get('epoch', 0)
        return start_epoch, None

    def validate_one_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(':')[0]

        with torch.no_grad():
            for x, y in self.vallo:
                x = x.to(self.device)
                y = y.to(self.device)
                weights = self.lat_weight(y.shape)
                with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                    x_recon, mu, log_var = self.model(x)
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
            self.writer.add_scalar("Loss/val/kl_scaled", avg_kl * self.beta, global_step)

        return avg_loss

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        pbar = tqdm(self.trainlo, desc=f"Epoch {epoch+1}/{self.epochs}")

        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(':')[0]

        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(self.device)
            y = y.to(self.device)

            weights = self.lat_weight(y.shape)

            self.opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                x_recon, mu, log_var = self.model(x)
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
        self.writer.add_scalar("Loss/train/kl_scaled", avg_kl * self.beta, global_step)
        self.writer.add_scalar("hyper/beta",          self.beta,  global_step)
        self.writer.add_scalar("hyper/lr",            self.opt.param_groups[0]['lr'], global_step)


        return avg_loss


    def train(self, resume_path=None, only_model = False):
        start_epoch = 0
        best_metric = None
        
        if resume_path is not None:
            start_epoch, best_metric = self.load_checkpoint(resume_path, strict=True, only_model = only_model)

        try:

            for epoch in range(start_epoch, self.epochs):
                avg_loss = self.train_one_epoch(epoch)
                val_loss = self.validate_one_epoch(epoch)
                if (epoch + 1) % self.save_interval == 0:
                    self.save_checkpoint(epoch, avg_loss)

        finally:
            if hasattr(self, 'writer') and self.writer:
                self.writer.close()        
                print("TensorBoard writer closed.")


    

