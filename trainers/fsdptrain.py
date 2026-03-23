import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import ReduceLROnPlateau

from trainers.trainUNET import UNetTrainer


class FSDPUNetTrainer(UNetTrainer):
    """UNetTrainer 的 FSDP 版本。

    - 支持单机多卡 FSDP 训练
    - 只在 rank==0 时写日志 / 保存模型 / 打印 epoch 级信息
    - 对 DistributedSampler 调用 set_epoch
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer,
        scheduler,
        epochs: int,
        device: torch.device,
        beta: float = 0.0,
        tb_dir: str = "./tensorboard_logs",
        save_dir: str = "./checkpoints",
        save_interval: int = 1,
        use_amp: bool = False,
        rank: int = 0,
        world_size: int = 1,
        is_master: bool | None = None,
        kl_anneal: bool = False,
        kl_anneal_epochs: int = 10,
    ):
        self.rank = rank
        self.world_size = world_size
        self.is_master = (rank == 0) if is_master is None else is_master

        # 对 FSDP 包裹的模型，同步内外层的 using_kl 标志，
        # 确保 Trainer 在分布式场景下也能正确识别是否启用 KL
        inner_using_kl = False
        if hasattr(model, "using_kl"):
            inner_using_kl = bool(getattr(model, "using_kl", False))
        elif hasattr(model, "module") and hasattr(model.module, "using_kl"):
            inner_using_kl = bool(getattr(model.module, "using_kl", False))

        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            device=device,
            beta=beta,
            tb_dir=tb_dir,
            save_dir=save_dir,
            save_interval=save_interval,
            use_amp=use_amp,
            kl_anneal=kl_anneal,
            kl_anneal_epochs=kl_anneal_epochs,
        )

        # 覆盖/刷新 Trainer 自身的 using_kl 标志
        # （单卡时 UNetTrainer 已在 __init__ 中设置，这里在 FSDP 场景下做一次统一）
        self.using_kl = bool(getattr(self, "using_kl", False) or inner_using_kl)

        if self.is_master:
            print(f"[FSDPUNetTrainer] using_kl = {self.using_kl}")

        # 非主进程关闭 TensorBoard，避免多进程同时写
        if not self.is_master and hasattr(self, "writer") and self.writer is not None:
            self.writer.close()
            self.writer = None

    def save_checkpoint(self, epoch, current_avg_loss):
        if not self.is_master:
            return
        super().save_checkpoint(epoch, current_avg_loss)

    def _all_reduce_loss(self, total_loss: float, total_recon: float, num_batches: int):
        """在所有进程间做 all_reduce，得到全局平均 loss。"""
        if not dist.is_available() or not dist.is_initialized():
            avg_loss = total_loss / max(num_batches, 1)
            avg_recon = total_recon / max(num_batches, 1)
            return avg_loss, avg_recon

        device = self.device if isinstance(self.device, torch.device) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor([total_loss, total_recon, float(num_batches)], device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_loss_g, total_recon_g, num_batches_g = tensor.tolist()
        num_batches_g = max(num_batches_g, 1.0)
        avg_loss = float(total_loss_g / num_batches_g)
        avg_recon = float(total_recon_g / num_batches_g)
        return avg_loss, avg_recon

    def validate_one_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        num_batches = 0

        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(":")[0]

        with torch.no_grad():
            for batch_idx, (x, y, i, times) in enumerate(self.vallo):
                x = x.to(self.device)
                y = y.to(self.device)

                # 检查验证集 batch 是否存在 NaN/Inf，并打印对应时间
                has_nan_inf_x = torch.isnan(x).any() or torch.isinf(x).any()
                has_nan_inf_y = torch.isnan(y).any() or torch.isinf(y).any()
                if has_nan_inf_x or has_nan_inf_y:
                    if self.is_master:
                        times_str = ", ".join(str(t) for t in list(times))
                        print(f"[Val] batch {batch_idx} contains NaN/Inf, times: {times_str}")
                        print("[Val] 该 batch 已跳过，用于避免验证损失变为 NaN")
                    continue

                # i: lead time index，这里不再参与模型计算
                weights = self.lat_weight(y.shape)
                with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                    if getattr(self, "using_kl", False):
                        x_recon, mu, log_var = self.model(x, times=times)
                    else:
                        x_recon = self.model(x, times=times)
                        mu = log_var = None

                    recon_loss = self.cal_losses(x_recon, y, weight=weights)

                    if getattr(self, "using_kl", False) and mu is not None and log_var is not None:
                        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                        loss = recon_loss + self.beta * kl_loss
                    else:
                        loss = recon_loss

                total_loss += float(loss.detach())
                total_recon_loss += float(recon_loss.detach())
                num_batches += 1

        # 若全部 batch 都被跳过，避免除以 0
        if num_batches == 0:
            if self.is_master:
                print("[Val] 所有 batch 均因包含 NaN/Inf 被跳过，返回损失 0.0 以保持训练继续进行")
            avg_loss = 0.0
            avg_recon = 0.0
        else:
            avg_loss, avg_recon = self._all_reduce_loss(total_loss, total_recon_loss, num_batches)

        if self.is_master:
            print(f"\nEpoch {epoch+1} 验证集平均:")
            # 这里暂时只打印总损失和重建损失，KL 如需可在上面累加并打印
            print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}")

            global_step = epoch
            if hasattr(self, "writer") and self.writer:
                self.writer.add_scalar("Loss/val/total", avg_loss, global_step)
                self.writer.add_scalar("Loss/val/recon", avg_recon, global_step)

        return avg_loss

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        # DistributedSampler 设 epoch，保证每轮 shuffle 不同
        sampler = getattr(self.trainlo, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        from tqdm import tqdm

        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(":")[0]

        pbar = tqdm(self.trainlo, desc=f"Epoch {epoch+1}/{self.epochs}", disable=not self.is_master)

        for batch_idx, (x, y, i, times) in enumerate(pbar):
            x = x.to(self.device)
            y = y.to(self.device)
            # i: lead time index，这里不再参与模型计算
            if torch.isnan(x).any() or torch.isinf(x).any():
                if self.is_master:
                    print(f"Batch {batch_idx} input contains nan/inf!")
            if torch.isnan(y).any() or torch.isinf(y).any():
                if self.is_master:
                    print(f"Batch {batch_idx} target contains nan/inf!")

            weights = self.lat_weight(y.shape)

            self.opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                if getattr(self, "using_kl", False):
                    x_recon, mu, log_var = self.model(x, times=times)
                else:
                    x_recon = self.model(x, times=times)
                    mu = log_var = None

                recon_loss = self.cal_losses(x_recon, y, weight=weights)

                if getattr(self, "using_kl", False) and mu is not None and log_var is not None:
                    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                    loss = recon_loss + self.beta * kl_loss
                else:
                    kl_loss = torch.tensor(0.0, device=self.device)
                    loss = recon_loss

            loss_item = float(loss.detach())
            recon_item = float(recon_loss.detach())
            kl_item = float(kl_loss.detach())

            total_loss += loss_item
            total_recon_loss += recon_item
            total_kl_loss += kl_item

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)

            # FSDP 建议使用专门的梯度裁剪：传入 FSDP 模型本身
            if isinstance(self.model, FSDP):
                FSDP.clip_grad_norm_(self.model, max_norm=5.0)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.scaler.step(self.opt)
            self.scaler.update()

            if self.is_master:
                if getattr(self, "using_kl", False):
                    pbar.set_postfix({
                        'Loss': f'{loss_item:.4f}',
                        'Recon': f'{recon_item:.4f}',
                        'KL': f'{kl_item:.4f}',
                    })
                else:
                    pbar.set_postfix({
                        'Loss': f'{loss_item:.4f}',
                        'Recon': f'{recon_item:.4f}',
                    })

                if batch_idx % 10 == 0 and hasattr(self, 'writer') and self.writer:
                    step = epoch * len(self.trainlo) + batch_idx
                    self.writer.add_scalar("Loss/batch/total", loss_item, step)
                    self.writer.add_scalar("Loss/batch/recon", recon_item, step)
                    if getattr(self, "using_kl", False):
                        self.writer.add_scalar("Loss/batch/kl", kl_item, step)

        # 得到全局平均 train loss（这里只 all_reduce 总损失和重建损失，KL 已体现在总损失中）
        avg_loss, avg_recon = self._all_reduce_loss(total_loss, total_recon_loss, len(self.trainlo))

        # 验证也返回全局损失
        val_loss = self.validate_one_epoch(epoch)

        # scheduler 所有 rank 都要 step，保证 lr 一致
        if isinstance(self.sch, ReduceLROnPlateau):
            self.sch.step(val_loss)
        else:
            self.sch.step()

        if self.is_master:
            print(f"\nEpoch {epoch+1} 训练集平均:")
            print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}")

            global_step = epoch
            if hasattr(self, 'writer') and self.writer:
                self.writer.add_scalar("Loss/train/total",    avg_loss,  global_step)
                self.writer.add_scalar("Loss/train/recon",    avg_recon, global_step)
                # 当使用 KL 且启用 KL annealing 时，记录当前 beta
                if getattr(self, "using_kl", False) and getattr(self, "kl_anneal", False):
                    self.writer.add_scalar("hyper/beta",      self.beta, global_step)
                self.writer.add_scalar("hyper/lr",            self.opt.param_groups[0]['lr'], global_step)

        return avg_loss, val_loss
