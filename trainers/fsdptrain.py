import os
from pathlib import Path
from functools import partial
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.checkpoint import checkpoint_sequential
from tqdm import tqdm
from .basetrain import BaseTrainer

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
        min_num_params: int = 1e5,
        optimizer_config: dict = None,
        scheduler_config: dict = None,
        use_activation_ckpt: bool = False,
    ):
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
            use_fsdp = False  # 单机不包 FSDP

        model = model.to(device)

        if use_fsdp and self.is_distributed:
            model = self._wrap_fsdp(
                model,
                sharding_strategy,
                mixed_precision,
                min_num_params,
                use_activation_ckpt,
            )

        if optimizer is None and optimizer_config is not None:
            optimizer = self._create_optimizer(model, optimizer_config)
        if scheduler is None and scheduler_config is not None:
            scheduler = self._create_scheduler(optimizer, scheduler_config)

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

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir)) if self.rank == 0 else None
        self.global_step = 0

        if self.rank == 0:
            print("✅ FSDPTrainer 初始化完成")
            print(f"   分布式: {self.is_distributed}, GPU 数: {self.world_size}")
            print(f"   分片策略: {sharding_strategy}, 混合精度: {mixed_precision}")

    def _wrap_fsdp(self, model, sharding_strategy, mixed_precision, min_num_params, use_activation_ckpt):
        strategy = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=min_num_params)

        mp_policy = None
        if mixed_precision:
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )

        fsdp_model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=strategy,
            mixed_precision=mp_policy,
            device_id=self.local_rank,
        )

        # 可选：激活检查点（节省激活显存）
        if use_activation_ckpt:
            for name, module in fsdp_model.named_modules():
                if isinstance(module, nn.Sequential) and len(module) > 1:
                    # 这里示例性对较长的 sequential 做 checkpoint
                    module.forward = checkpoint_sequential(module, chunks=2)

        if self.rank == 0:
            print("✅ FSDP 包装完成")

        return fsdp_model

    def _create_optimizer(self, model, cfg):
        opt_type = cfg.get("type", "Adam")
        lr = cfg.get("lr", 1e-4)
        wd = cfg.get("weight_decay", 1e-5)
        if opt_type.lower() == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=cfg.get("betas", (0.9, 0.999)))
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=cfg.get("betas", (0.9, 0.999)))

    def _create_scheduler(self, optimizer, cfg):
        if optimizer is None or cfg is None:
            return None
        st = cfg.get("type", "CosineAnnealingLR")
        if st == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.get("T_max", 30), eta_min=cfg.get("eta_min", 1e-6)
            )
        if st == "StepLR":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg.get("step_size", 10), gamma=cfg.get("gamma", 0.1)
            )
        return None

    def _set_sampler_epoch(self, epoch):
        if hasattr(self.trainlo, "sampler") and hasattr(self.trainlo.sampler, "set_epoch"):
            self.trainlo.sampler.set_epoch(epoch)

    def train_one_epoch(self, epoch):
        self._set_sampler_epoch(epoch)
        self.model.train()
        total_loss = total_recon = total_kl = 0.0

        pbar = tqdm(self.trainlo, desc=f"Epoch {epoch+1}/{self.epochs}", disable=self.rank != 0)
        for batch_idx, (x, y, t) in enumerate(pbar):
            x = x.unsqueeze(2).to(self.device)
            y = y.to(self.device)

            x_recon, mu, log_var = self.model(x)
            kl_loss, recon_loss = self.cal_losses(x_recon, y, mu, log_var)
            loss = kl_loss * self.beta + recon_loss

            if not torch.isfinite(loss):
                if self.rank == 0:
                    print(f"❌ NaN/Inf @ batch {batch_idx}: kl={kl_loss.item():.4f}, recon={recon_loss.item():.4f}")
                continue

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            self.global_step += 1

            pbar.set_postfix(Loss=f"{loss.item():.4f}", Recon=f"{recon_loss.item():.4f}", KL=f"{kl_loss.item():.4f}")

            if self.writer:
                self.writer.add_scalar("Loss/batch", loss.item(), self.global_step)
                self.writer.add_scalar("Recon/batch", recon_loss.item(), self.global_step)
                self.writer.add_scalar("KL/batch", kl_loss.item(), self.global_step)

        if self.sch:
            self.sch.step()

        n = len(self.trainlo)
        avg_loss, avg_recon, avg_kl = total_loss / n, total_recon / n, total_kl / n
        if self.rank == 0:
            print(f"\nEpoch {epoch+1} 训练集平均: 总损失={avg_loss:.5f}, 重建={avg_recon:.5f}, 散度={avg_kl:.5f}")
            if self.writer:
                self.writer.add_scalar("Loss/epoch", avg_loss, epoch)
                self.writer.add_scalar("Recon/epoch", avg_recon, epoch)
                self.writer.add_scalar("KL/epoch", avg_kl, epoch)
                if self.sch:
                    self.writer.add_scalar("LR", self.sch.get_last_lr()[0], epoch)
        return avg_loss, avg_recon, avg_kl

    def eval_one_epoch(self):
        self.model.eval()
        total_loss = total_recon = total_kl = 0.0
        with torch.no_grad():
            pbar = tqdm(self.testlo, desc="Evaluating", disable=self.rank != 0)
            for x, y, t in pbar:
                x = x.unsqueeze(2).to(self.device)
                y = y.to(self.device)
                x_recon, mu, log_var = self.model(x)
                kl_loss, recon_loss = self.cal_losses(x_recon, y, mu, log_var)
                loss = kl_loss * self.beta + recon_loss
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                pbar.set_postfix(Loss=f"{loss.item():.4f}", Recon=f"{recon_loss.item():.4f}", KL=f"{kl_loss.item():.4f}")

        n = len(self.testlo)
        avg_loss, avg_recon, avg_kl = total_loss / n, total_recon / n, total_kl / n
        if self.rank == 0:
            print(f"\n验证集平均: 总损失={avg_loss:.5f}, 重建={avg_recon:.5f}, 散度={avg_kl:.5f}")
            if self.writer:
                self.writer.add_scalar("Val/Loss", avg_loss, self.global_step)
                self.writer.add_scalar("Val/Recon", avg_recon, self.global_step)
                self.writer.add_scalar("Val/KL", avg_kl, self.global_step)
        return avg_loss, avg_recon, avg_kl

    def train(self, resume_path=None):
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
            if self.writer:
                self.writer.close()
                if self.rank == 0:
                    print(f"\n✅ TensorBoard 日志已保存到: {self.log_dir}")
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()
                if self.rank == 0:
                    print("✅ FSDP 进程组已清理")

    def save_checkpoint(self, epoch, tag="last", best_metric=None):
        if self.rank != 0:
            return None
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state = self.model.state_dict()
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.opt.state_dict() if self.opt else None,
            "scheduler_state_dict": self.sch.state_dict() if self.sch else None,
            "best_metric": best_metric,
        }
        save_path = self.save_dir / f"epoch_{epoch+1}_{tag}.pt"
        torch.save(ckpt, save_path)
        print(f"✅ FSDP Checkpoint 已保存: {save_path}")
        return str(save_path)

    def load_checkpoint(self, path, strict=True, load_optim=True, load_sched=True):
        if not Path(path).exists():
            raise FileNotFoundError(f"❌ Checkpoint 不存在: {path}")
        ckpt = torch.load(path, map_location="cpu")
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, load_policy):
            self.model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        if load_optim and ckpt.get("optimizer_state_dict") and self.opt:
            self.opt.load_state_dict(ckpt["optimizer_state_dict"])
        if load_sched and ckpt.get("scheduler_state_dict") and self.sch:
            self.sch.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_metric = ckpt.get("best_metric")
        if self.rank == 0:
            print(f"✅ 已加载 FSDP checkpoint: {path}, 下一轮从 epoch {start_epoch} 开始")
        return start_epoch, best_metric