import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import torch.distributed as dist

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
        
        # checkpoint 参数
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval

    def cal_losses(self, x, y, mu, logvar):
        """计算 KL 散度和重建损失"""
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = torch.mean((x - y) ** 2)
        return kl_loss, recon_loss

    def _model_state_dict(self):
        """获取模型权重，兼容 DDP/单卡"""
        return self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()

    def _load_model_state_dict(self, sd, strict=True):
        """加载模型权重，兼容 DDP/单卡"""
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(sd, strict=strict)
        else:
            self.model.load_state_dict(sd, strict=strict)

    def save_checkpoint(self, epoch, tag="last", best_metric=None):
        """保存 checkpoint（包含模型、优化器、调度器状态）"""
        # DDP 仅在 rank0 保存
        if dist.is_initialized() and dist.get_rank() != 0:
            return None
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self._model_state_dict(),
            'optimizer_state_dict': self.opt.state_dict() if self.opt else None,
            'scheduler_state_dict': self.sch.state_dict() if self.sch else None,
            'best_metric': best_metric,
        }
        save_path = self.save_dir / f"epoch_{epoch+1}_{tag}.pt"
        torch.save(checkpoint, save_path)
        print(f"✅ Checkpoint 已保存: {save_path}")
        return str(save_path)

    def load_checkpoint(self, path, strict=True, load_optim=True, load_sched=True):
        """
        加载 checkpoint（包含模型、优化器、调度器状态）
        
        Returns:
            tuple: (start_epoch, best_metric)
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"❌ Checkpoint 不存在: {path}")
        
        ckpt = torch.load(path, map_location="cpu")
        
        # 加载模型权重
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            self._load_model_state_dict(ckpt['model_state_dict'], strict=strict)
        else:
            # 向后兼容：直接作为 state_dict
            self._load_model_state_dict(ckpt, strict=strict)
        
        # 加载优化器状态
        if load_optim and ckpt.get('optimizer_state_dict') and self.opt is not None:
            self.opt.load_state_dict(ckpt['optimizer_state_dict'])
        
        # 加载调度器状态
        if load_sched and ckpt.get('scheduler_state_dict') and self.sch is not None:
            self.sch.load_state_dict(ckpt['scheduler_state_dict'])
        
        start_epoch = ckpt.get('epoch', -1) + 1
        best_metric = ckpt.get('best_metric')
        
        print(f"✅ 已加载 checkpoint: {path}")
        print(f"   严格模式: {strict}")
        print(f"   恢复优化器: {load_optim}")
        print(f"   恢复调度器: {load_sched}")
        print(f"   下一轮从 epoch {start_epoch} 开始")
        if best_metric is not None:
            print(f"   最佳指标: {best_metric}")
        
        return start_epoch, best_metric

    def train_one_epoch(self, epoch):
        """单轮训练"""
        self.model.train()
        total_loss = 0.0  # ✅ 用 float，不是 tensor
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        pbar = tqdm(self.trainlo, desc=f"Epoch {epoch+1}/{self.epochs}")

        for batch_idx, (x, y, t) in enumerate(pbar):
            x = x.unsqueeze(2).to(self.device)
            y = y.to(self.device)

            x_recon, mu, log_var = self.model(x)
            kl_loss, recon_loss = self.cal_losses(x_recon, y, mu, log_var)
            loss = kl_loss * self.beta + recon_loss

            # ✅ 用 .item() 转换为 float
            loss_item = loss.item()
            recon_item = recon_loss.item()
            kl_item = kl_loss.item()

            total_loss += loss_item
            total_recon_loss += recon_item
            total_kl_loss += kl_item

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()

            pbar.set_postfix({
                'Loss': f'{loss_item:.4f}',
                'Recon': f'{recon_item:.4f}',
                'KL': f'{kl_item:.4f}',
            })

        self.sch.step()

        # ✅ 现在都是 float，计算无问题
        avg_loss = total_loss / len(self.trainlo)
        avg_recon = total_recon_loss / len(self.trainlo)
        avg_kl = total_kl_loss / len(self.trainlo)
        
        print(f"\nEpoch {epoch+1} 训练集平均:")
        print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}, 散度={avg_kl:.5f}")

    def train(self, resume_path=None):
        """训练循环，支持断点续传"""
        start_epoch = 0
        best_metric = None
        
        if resume_path is not None:
            start_epoch, best_metric = self.load_checkpoint(resume_path, strict=True)

        for epoch in range(start_epoch, self.epochs):
            self.train_one_epoch(epoch)
            
            # 间隔保存
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch, tag="last", best_metric=best_metric)