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
        # ✅ 新增：优化器和调度器的配置参数
        optimizer_config: dict = None,
        scheduler_config: dict = None,
    ):
        # ✅ 暂存优化器/调度器配置（在 FSDP 包装前）
        if optimizer is not None:
            self.optimizer_config = {
                'type': type(optimizer).__name__,
                'lr': optimizer.defaults.get('lr', 1e-4),
                'weight_decay': optimizer.defaults.get('weight_decay', 1e-5),
                'betas': optimizer.defaults.get('betas', (0.9, 0.999)),
            }
        else:
            self.optimizer_config = optimizer_config or {
                'type': 'Adam',
                'lr': 1e-4,
                'weight_decay': 1e-5,
                'betas': (0.9, 0.999),
            }
        
        if scheduler is not None:
            if hasattr(scheduler, 'T_max'):
                self.scheduler_config = {
                    'type': 'CosineAnnealingLR',
                    'T_max': scheduler.T_max,
                    'eta_min': getattr(scheduler, 'eta_min', 1e-6),
                }
            else:
                self.scheduler_config = {'type': None}
        else:
            self.scheduler_config = scheduler_config or {
                'type': 'CosineAnnealingLR',
                'T_max': 30,
                'eta_min': 1e-6,
            }
        
        # ✅ 先传递 None 给父类（稍后重新创建）
        super().__init__(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=None,  # ✅ 暂时传 None
            scheduler=None,  # ✅ 暂时传 None
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
        
        # ✅ FSDP 包装后创建优化器和调度器
        self._setup_optimizer_scheduler()
    
    def _setup_fsdp(self):
        """初始化 FSDP 模型包装"""
        torch.cuda.set_device(self.local_rank)
        # ✅ 只移动到对应的 GPU，不调用 .to(device)
        # FSDP 会自动处理设备分配
        if not next(self.model.parameters()).is_cuda:
            self.model = self.model.to(self.local_rank)
        
        # 设置分片策略
        sharding_strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }
        strategy = sharding_strategy_map.get(self.sharding_strategy, ShardingStrategy.FULL_SHARD)
        
        # 自动分片策略
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=self.min_num_params
        )
        
        # 混合精度配置
        mixed_precision_policy = None
        if self.mixed_precision:
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        
        # 包装 FSDP
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
            
            # ✅ 打印参数量（验证分片）
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"   本 GPU 参数量: {total_params / 1e6:.2f}M")
    
    def _setup_optimizer_scheduler(self):
        """✅ 在 FSDP 包装后创建优化器和调度器"""
        # 创建优化器
        opt_cfg = self.optimizer_config
        if opt_cfg['type'] == 'Adam':
            self.opt = torch.optim.Adam(
                self.model.parameters(),
                lr=opt_cfg['lr'],
                weight_decay=opt_cfg['weight_decay'],
                betas=opt_cfg['betas']
            )
        elif opt_cfg['type'] == 'AdamW':
            self.opt = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_cfg['lr'],
                weight_decay=opt_cfg['weight_decay'],
                betas=opt_cfg['betas']
            )
        elif opt_cfg['type'] == 'SGD':
            self.opt = torch.optim.SGD(
                self.model.parameters(),
                lr=opt_cfg['lr'],
                weight_decay=opt_cfg['weight_decay'],
                momentum=opt_cfg.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"不支持的优化器类型: {opt_cfg['type']}")
        
        # 创建调度器
        sch_cfg = self.scheduler_config
        if sch_cfg.get('type') == 'CosineAnnealingLR':
            self.sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt,
                T_max=sch_cfg['T_max'],
                eta_min=sch_cfg['eta_min']
            )
        elif sch_cfg.get('type') == 'StepLR':
            self.sch = torch.optim.lr_scheduler.StepLR(
                self.opt,
                step_size=sch_cfg.get('step_size', 10),
                gamma=sch_cfg.get('gamma', 0.1)
            )
        elif sch_cfg.get('type') is None:
            self.sch = None
        else:
            raise ValueError(f"不支持的调度器类型: {sch_cfg['type']}")
        
        if self.rank == 0:
            print(f"✅ 优化器已创建: {opt_cfg['type']} (lr={opt_cfg['lr']})")
            if self.sch is not None:
                print(f"✅ 调度器已创建: {sch_cfg['type']}")
    
    def _set_sampler_epoch(self, epoch):
        """设置 DistributedSampler 的 epoch，保证随机性"""
        if hasattr(self.trainlo, 'sampler') and isinstance(
            self.trainlo.sampler, torch.utils.data.distributed.DistributedSampler
        ):
            self.trainlo.sampler.set_epoch(epoch)
    
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