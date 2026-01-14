from .basetrain import BaseTrainer
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import os
import torch.distributed as dist
from functools import partial

class FSDPTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        epochs,
        device,
        beta,
        log_dir: str = "./logs",
        use_fsdp: bool = True,
        save_dir: str = "./checkpoints",
        save_interval: int = 10,
        sharding_strategy: str = "FULL_SHARD",  # ✅ 可选：FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
        mixed_precision: bool = False,  # ✅ 是否使用混合精度
        min_num_params: int = 1e6,  # ✅ 自动分片的最小参数量
    ):
        # ✅ 显式传递所有参数给父类
        super().__init__(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            device=device,
            beta=beta,
            save_dir=save_dir,
            save_interval=save_interval,
        )
        
        # 检查是否通过 torchrun 启动
        self.is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        
        if self.is_distributed:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
                if self.rank == 0:
                    print(f"✅ FSDP 进程组已初始化：{self.world_size} 张 GPU")
            
            torch.cuda.set_device(self.local_rank)
            # ✅ 重新设置 device 为 local_rank
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.use_fsdp = use_fsdp
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.use_fsdp = False
            if self.rank == 0:
                print(f"ℹ️  单卡训练模式")
        
        # TensorBoard（仅主进程）
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None
        
        self.global_step = 0
        
        # FSDP 配置
        self.sharding_strategy = sharding_strategy
        self.mixed_precision = mixed_precision
        self.min_num_params = min_num_params
        
        # 包装 FSDP
        if self.use_fsdp and self.is_distributed:
            self._setup_fsdp()
    
    def _setup_fsdp(self):
        """初始化 FSDP 模型包装"""
        torch.cuda.set_device(self.local_rank)
        self.model = self.model.to(self.local_rank)
        
        # ✅ 设置分片策略
        sharding_strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,  # 完全分片（最省显存）
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,  # 仅分片梯度
            "NO_SHARD": ShardingStrategy.NO_SHARD,  # 不分片（等同 DDP）
        }
        strategy = sharding_strategy_map.get(self.sharding_strategy, ShardingStrategy.FULL_SHARD)
        
        # ✅ 自动分片策略（基于参数量）
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=self.min_num_params
        )
        
        # ✅ 混合精度配置
        mixed_precision_policy = None
        if self.mixed_precision:
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        
        # ✅ 包装 FSDP
        self.model = FSDP(
            self.model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=strategy,
            mixed_precision=mixed_precision_policy,
            device_id=self.local_rank,
        )
        
        if self.rank == 0:
            print(f"✅ FSDP 模型已包装")
            print(f"   分片策略: {self.sharding_strategy}")
            print(f"   混合精度: {self.mixed_precision}")
            print(f"   使用 GPU 数: {self.world_size}")
    
    def _set_sampler_epoch(self, epoch):
        """设置 DistributedSampler 的 epoch，保证随机性"""
        if hasattr(self.trainlo, 'sampler') and isinstance(
            self.trainlo.sampler, torch.utils.data.distributed.DistributedSampler
        ):
            self.trainlo.sampler.set_epoch(epoch)
    
    def train_one_epoch(self, epoch):
        """单轮训练（FSDP 版本）"""
        # ✅ 设置 sampler epoch
        self._set_sampler_epoch(epoch)
        
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
        
        return avg_loss, avg_recon, avg_kl

    def eval_one_epoch(self):
        """验证集评估（FSDP 版本）"""
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(
                self.testlo,
                desc="Evaluating",
                disable=self.rank != 0,
            )
            
            for batch_idx, (x, y, t) in enumerate(pbar):
                x = x.unsqueeze(2).to(self.device)
                y = y.to(self.device)

                x_recon, mu, log_var = self.model(x)
                kl_loss, recon_loss = self.cal_losses(x_recon, y, mu, log_var)
                loss = kl_loss * self.beta + recon_loss

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Recon': f'{recon_loss.item():.4f}',
                    'KL': f'{kl_loss.item():.4f}',
                })

        avg_loss = total_loss / len(self.testlo)
        avg_recon = total_recon_loss / len(self.testlo)
        avg_kl = total_kl_loss / len(self.testlo)
        
        if self.rank == 0:
            print(f"\n验证集平均:")
            print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}, 散度={avg_kl:.5f}")
            
            if self.writer is not None:
                self.writer.add_scalar('Val/Loss', avg_loss, self.global_step)
                self.writer.add_scalar('Val/ReconLoss', avg_recon, self.global_step)
                self.writer.add_scalar('Val/KLLoss', avg_kl, self.global_step)
        
        return avg_loss, avg_recon, avg_kl

    def train(self, resume_path=None):
        """训练循环，支持断点续传和验证"""
        start_epoch = 0
        best_metric = None
        
        # ✅ 从 checkpoint 恢复
        if resume_path is not None:
            start_epoch, best_metric = self.load_checkpoint(resume_path, strict=True)
        
        try:
            for epoch in range(start_epoch, self.epochs):
                # 训练
                self.train_one_epoch(epoch)
                
                
                
                # 保存 checkpoint
                if (epoch + 1) % self.save_interval == 0:
                    self.save_checkpoint(epoch, tag="last", best_metric=best_metric)
        
        finally:
            if self.writer is not None:
                self.writer.close()
                if self.rank == 0:
                    print(f"\n✅ TensorBoard 日志已保存到: {self.log_dir}")
            
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()
                if self.rank == 0:
                    print("✅ FSDP 进程组已清理")
    
    def save_checkpoint(self, epoch, tag="last", best_metric=None):
        """保存 FSDP checkpoint（需要特殊处理）"""
        if self.rank != 0:
            return None
        
        # ✅ FSDP 需要先聚合所有分片
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(
            self.model, StateDictType.FULL_STATE_DICT, save_policy
        ):
            model_state = self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.opt.state_dict() if self.opt else None,
            'scheduler_state_dict': self.sch.state_dict() if self.sch else None,
            'best_metric': best_metric,
        }
        
        save_path = self.save_dir / f"epoch_{epoch+1}_{tag}.pt"
        torch.save(checkpoint, save_path)
        print(f"✅ FSDP Checkpoint 已保存: {save_path}")
        return str(save_path)
    
    def load_checkpoint(self, path, strict=True, load_optim=True, load_sched=True):
        """加载 FSDP checkpoint"""
        if not Path(path).exists():
            raise FileNotFoundError(f"❌ Checkpoint 不存在: {path}")
        
        ckpt = torch.load(path, map_location="cpu")
        
        # ✅ FSDP 需要特殊的加载方式
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(
            self.model, StateDictType.FULL_STATE_DICT, load_policy
        ):
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                self.model.load_state_dict(ckpt['model_state_dict'], strict=strict)
            else:
                self.model.load_state_dict(ckpt, strict=strict)
        
        # 加载优化器状态
        if load_optim and ckpt.get('optimizer_state_dict') and self.opt is not None:
            self.opt.load_state_dict(ckpt['optimizer_state_dict'])
        
        # 加载调度器状态
        if load_sched and ckpt.get('scheduler_state_dict') and self.sch is not None:
            self.sch.load_state_dict(ckpt['scheduler_state_dict'])
        
        start_epoch = ckpt.get('epoch', -1) + 1
        best_metric = ckpt.get('best_metric')
        
        if self.rank == 0:
            print(f"✅ 已加载 FSDP checkpoint: {path}")
            print(f"   严格模式: {strict}")
            print(f"   恢复优化器: {load_optim}")
            print(f"   恢复调度器: {load_sched}")
            print(f"   下一轮从 epoch {start_epoch} 开始")
            if best_metric is not None:
                print(f"   最佳指标: {best_metric}")
        
        return start_epoch, best_metric