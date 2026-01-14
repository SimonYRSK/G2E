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
        sharding_strategy: str = "FULL_SHARD",
        mixed_precision: bool = False,
        min_num_params: int = 1e6,
        optimizer_config: dict = None,  # ✅ 新增
        scheduler_config: dict = None,  # ✅ 新增
    ):
        # ✅ 步骤 1: 初始化分布式环境
        self.is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        
        if self.is_distributed:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            
            torch.cuda.set_device(self.local_rank)
            device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.is_distributed = False
        
        # ✅ 步骤 2: 将模型移到 device
        model = model.to(device)
        
        # ✅ 步骤 3: FSDP 包装（必须在优化器创建前）
        if use_fsdp and self.is_distributed:
            model = self._wrap_fsdp(model, sharding_strategy, mixed_precision, min_num_params)
        
        # ✅ 步骤 4: 创建优化器
        if optimizer is None and optimizer_config is not None:
            optimizer = self._create_optimizer(model, optimizer_config)
        
        # ✅ 步骤 5: 创建调度器
        if scheduler is None and scheduler_config is not None:
            scheduler = self._create_scheduler(optimizer, scheduler_config)
        
        # ✅ 步骤 6: 调用父类初始化
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
        
        # TensorBoard
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None
        
        self.global_step = 0
        
        # FSDP 配置存储
        self.sharding_strategy = sharding_strategy
        self.mixed_precision = mixed_precision
        self.min_num_params = min_num_params
        
        if self.rank == 0:
            print(f"✅ FSDPTrainer 初始化完成")
            if self.is_distributed:
                print(f"   分布式: 是，{self.world_size} 张 GPU")
                print(f"   分片策略: {sharding_strategy}")
            else:
                print(f"   分布式: 否，单卡模式")
    
    def _wrap_fsdp(self, model, sharding_strategy, mixed_precision, min_num_params):
        """包装模型为 FSDP"""
        sharding_strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }
        strategy = sharding_strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)
        
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
        
        mixed_precision_policy = None
        if mixed_precision:
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=strategy,
            mixed_precision=mixed_precision_policy,
            device_id=self.local_rank,
        )
        
        if self.rank == 0:
            print(f"✅ FSDP 包装完成")
        
        return model
    
    def _create_optimizer(self, model, optimizer_config):
        """创建优化器"""
        if optimizer_config is None:
            return None
        
        opt_type = optimizer_config.get('type', 'Adam')
        lr = optimizer_config.get('lr', 1e-4)
        
        if opt_type == 'Adam':
            return torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=optimizer_config.get('weight_decay', 1e-5),
                betas=optimizer_config.get('betas', (0.9, 0.999)),
            )
        elif opt_type == 'AdamW':
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=optimizer_config.get('weight_decay', 1e-5),
            )
        else:
            raise ValueError(f"不支持的优化器: {opt_type}")
    
    def _create_scheduler(self, optimizer, scheduler_config):
        """创建学习率调度器"""
        if scheduler_config is None or optimizer is None:
            return None
        
        sched_type = scheduler_config.get('type', 'CosineAnnealingLR')
        
        if sched_type == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get('T_max', 30),
                eta_min=scheduler_config.get('eta_min', 1e-6),
            )
        elif sched_type == 'StepLR':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1),
            )
        else:
            raise ValueError(f"不支持的调度器: {sched_type}")
    
    def _set_sampler_epoch(self, epoch):
        """设置 DistributedSampler 的 epoch"""
        if hasattr(self.trainlo, 'sampler') and isinstance(
            self.trainlo.sampler, torch.utils.data.distributed.DistributedSampler
        ):
            self.trainlo.sampler.set_epoch(epoch)
    
    # ...existing code...
    
    def train_one_epoch(self, epoch):
        """单轮训练（FSDP 版本）"""
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

        if self.sch is not None:
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
                if self.sch is not None:
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
        
        if resume_path is not None:
            start_epoch, best_metric = self.load_checkpoint(resume_path, strict=True)
        
        try:
            for epoch in range(start_epoch, self.epochs):
                self.train_one_epoch(epoch)
                
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
        
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(
            self.model, StateDictType.FULL_STATE_DICT, load_policy
        ):
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                self.model.load_state_dict(ckpt['model_state_dict'], strict=strict)
            else:
                self.model.load_state_dict(ckpt, strict=strict)
        
        if load_optim and ckpt.get('optimizer_state_dict') and self.opt is not None:
            self.opt.load_state_dict(ckpt['optimizer_state_dict'])
        
        if load_sched and ckpt.get('scheduler_state_dict') and self.sch is not None:
            self.sch.load_state_dict(ckpt['scheduler_state_dict'])
        
        start_epoch = ckpt.get('epoch', -1) + 1
        best_metric = ckpt.get('best_metric')
        
        if self.rank == 0:
            print(f"✅ 已加载 FSDP checkpoint: {path}")
            print(f"   下一轮从 epoch {start_epoch} 开始")
            if best_metric is not None:
                print(f"   最佳指标: {best_metric}")
        
        return start_epoch, best_metric