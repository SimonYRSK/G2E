from .basetrain import BaseTrainer
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import os
import torch.distributed as dist

class DDPTrainer(BaseTrainer):
    def __init__(
        self,
        log_dir: str = "./logs",
        use_ddp: bool = True,
        save_dir: str = "./checkpoints",
        save_interval: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs, save_dir=save_dir, save_interval=save_interval)
        
        # 检查是否通过 torchrun 启动（环境变量存在）
        self.is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        
        if self.is_distributed:
            # torchrun 启动：从环境变量读取
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            
            # 初始化进程组
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
                if self.rank == 0:
                    print(f"✅ DDP 进程组已初始化：{self.world_size} 张 GPU")
            
            torch.cuda.set_device(self.local_rank)
            self.use_ddp = use_ddp
        else:
            # 单卡启动：不使用 DDP
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.use_ddp = False
            if self.rank == 0:
                print(f"ℹ️  单卡训练模式（未检测到 torchrun）")
        
        # TensorBoard 配置（只在主进程）
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None
        
        self.global_step = 0
        
        # 如果启用 DDP，包装模型
        if self.use_ddp and self.is_distributed:
            self._setup_ddp()
    
    def _setup_ddp(self):
        """初始化 DDP 模型包装"""
        torch.cuda.set_device(self.local_rank)
        self.model = self.model.to(self.local_rank)
        
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False,
        )
        
        if self.rank == 0:
            print(f"✅ DDP 模型已包装，使用 {self.world_size} 张 GPU")
    
    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        pbar = tqdm(
            self.trainlo,
            desc=f"Epoch {epoch+1}/{self.epochs}",
            disable=self.rank != 0,
        )

        for batch_idx, (x, y, t) in enumerate(pbar):
            x = x.unsqueeze(2).to(self.device)
            y = y.to(self.device)

            x_recon, mu, log_var = self.model(x)
            kl_loss, recon_loss = self.cal_losses(x_recon, y, mu, log_var)
            loss = kl_loss * self.beta + recon_loss

            loss_item = float(loss.detach())
            recon_item = float(recon_loss.detach())
            kl_item = float(kl_loss.detach())

            total_loss += loss_item
            total_recon_loss += recon_item
            total_kl_loss += kl_item

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if self.writer is not None:
                self.writer.add_scalar('Loss/batch', loss_item, self.global_step)
                self.writer.add_scalar('ReconLoss/batch', recon_item, self.global_step)
                self.writer.add_scalar('KLLoss/batch', kl_item, self.global_step)
            
            self.global_step += 1

            pbar.set_postfix({
                'Loss': f'{loss_item:.4f}',
                'Recon': f'{recon_item:.4f}',
                'KL': f'{kl_item:.4f}',
            })

        self.sch.step()

        avg_loss = total_loss / len(self.trainlo)
        avg_recon = total_recon_loss / len(self.trainlo)
        avg_kl = total_kl_loss / len(self.trainlo)

        if self.rank == 0:
            print(f"\nEpoch {epoch+1} 训练集平均:")
            print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}, 散度={avg_kl:.5f}")
            
            if self.writer is not None:
                self.writer.add_scalar('Loss/epoch', avg_loss, epoch)
                self.writer.add_scalar('ReconLoss/epoch', avg_recon, epoch)
                self.writer.add_scalar('KLLoss/epoch', avg_kl, epoch)
                self.writer.add_scalar('LearningRate', self.sch.get_last_lr()[0], epoch)

    def save_checkpoint(self, epoch):
        """保存模型 checkpoint（只在主进程）"""
        if self.rank != 0:
            return
        
        # 如果是 DDP 模型，保存 module 的 state_dict
        model_state = self.model.module.state_dict() if self.use_ddp else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.opt.state_dict(),
            'scheduler_state_dict': self.sch.state_dict(),
        }
        
        save_path = self.save_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, save_path)
        print(f"✅ Checkpoint 已保存: {save_path}")

    def train(self):
        try:
            for epoch in range(self.epochs):
                self.train_one_epoch(epoch)
                
                # 只在主进程保存 checkpoint
                if self.rank == 0 and (epoch + 1) % self.save_interval == 0:
                    self.save_checkpoint(epoch)
        finally:
            if self.writer is not None:
                self.writer.close()
                print(f"\n✅ TensorBoard 日志已保存到: {self.log_dir}")
            
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()
                if self.rank == 0:
                    print("✅ DDP 进程组已清理")