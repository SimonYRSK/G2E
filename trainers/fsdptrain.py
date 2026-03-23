import os

import numpy as np
import pandas as pd
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
        plot_root: str | None = None,
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

        # 画图输出根目录（可由外部传入）
        self.plot_root = plot_root

        # 从验证集 Dataset 中记录通道名与经纬度，用于画图
        ds = getattr(val_loader, "dataset", None)
        self.plot_lat = None
        self.plot_lon = None
        self.channel_names = None
        self.channel_to_idx = None
        if ds is not None:
            # GFS2ERA5Dataset 中有 target_channels 和 ds_y
            if hasattr(ds, "target_channels"):
                self.channel_names = list(ds.target_channels)
                self.channel_to_idx = {name: idx for idx, name in enumerate(self.channel_names)}
            if hasattr(ds, "ds_y"):
                try:
                    self.plot_lat = ds.ds_y["lat"].values
                    self.plot_lon = ds.ds_y["lon"].values
                except Exception:
                    self.plot_lat = None
                    self.plot_lon = None

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

        # 只在第一个正常 batch 上画图
        has_plotted = False

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

                # 在首个正常 batch 上画图（只在主进程）
                if self.is_master and not has_plotted:
                    try:
                        self._plot_validation_maps(epoch, x_recon, y, times)
                    except Exception as e:
                        # 避免画图错误中断训练，仅在主进程打印
                        if self.is_master:
                            print(f"[Val] 绘图时出错: {e}")
                    has_plotted = True

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

    def _plot_validation_maps(self, epoch, x_recon, y, times):
        """在验证集首个正常 batch 上，为指定通道画 GT vs 预测 对比图。

        仅在 rank0 调用。借鉴 picture.py 的三联图格式：GT / Forecast / Forecast-GT，
        且 GT 与 Forecast 共用相同的 colorbar 范围。
        """
        if not self.is_master:
            return

        if self.plot_lat is None or self.plot_lon is None:
            print("[Val] 无法获取经纬度坐标，跳过画图")
            return

        if self.channel_to_idx is None or self.channel_names is None:
            print("[Val] 无法获取通道名称，跳过画图")
            return

        # 近地面变量与 500hPa 高空变量通道名
        near_surface_channels = ["t2m", "u10m", "v10m", "msl", "tp"]
        level500_channels = ["t500", "u500", "v500", "z500", "q500"]

        # 仅取当前 batch 的第一个样本作图
        pred_sample = x_recon[0].detach().cpu().numpy()  # (C, H, W)
        gt_sample = y[0].detach().cpu().numpy()          # (C, H, W)

        # 时间字符串用于文件名
        try:
            t0 = pd.Timestamp(str(times[0]))
            time_str = t0.strftime("%Y%m%d_%H%M")
        except Exception:
            time_str = "unknown_time"

        lat = self.plot_lat
        lon = self.plot_lon

        import matplotlib.pyplot as plt

        # 输出根目录：优先使用外部传入的 plot_root
        if self.plot_root is None:
            print("[Val] 未设置 plot_root，使用默认路径")
            out_root = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/channelpics/swinunet_2022_2024_3_21"
        else:
            out_root = self.plot_root
        epoch_dir = os.path.join(out_root, f"epoch_{epoch+1:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        def plot_one_channel(ch_name: str):
            if ch_name not in self.channel_to_idx:
                print(f"[Val] 通道 {ch_name} 不在当前 target_channels 中，跳过")
                return

            idx = self.channel_to_idx[ch_name]
            gt_2d = gt_sample[idx]
            pred_2d = pred_sample[idx]

            # 统一 GT 与 Forecast 的 colorbar 范围
            vmin = float(np.nanmin([gt_2d.min(), pred_2d.min()]))
            vmax = float(np.nanmax([gt_2d.max(), pred_2d.max()]))
            if vmin == vmax:
                vmax = vmin + 1e-6

            diff_2d = pred_2d - gt_2d
            diff_max = float(np.nanmax(np.abs(diff_2d)))
            if diff_max == 0:
                diff_max = 1e-6

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            im0 = axes[0].pcolormesh(lon, lat, gt_2d, shading="auto", vmin=vmin, vmax=vmax)
            axes[0].set_title(f"GT - {ch_name}")
            plt.colorbar(im0, ax=axes[0])

            im1 = axes[1].pcolormesh(lon, lat, pred_2d, shading="auto", vmin=vmin, vmax=vmax)
            axes[1].set_title(f"Forecast - {ch_name}")
            plt.colorbar(im1, ax=axes[1])

            im2 = axes[2].pcolormesh(lon, lat, diff_2d, shading="auto", vmin=-diff_max, vmax=diff_max)
            axes[2].set_title(f"Forecast - GT - {ch_name}")
            plt.colorbar(im2, ax=axes[2])

            for ax in axes:
                ax.set_xlabel("lon")
                ax.set_ylabel("lat")

            fig.suptitle(f"Epoch {epoch+1} Val Sample, {time_str}, {ch_name}")
            fig.tight_layout()

            fname = f"epoch{epoch+1:03d}_{time_str}_{ch_name}.png"
            save_path = os.path.join(epoch_dir, fname)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[Val] 已保存通道 {ch_name} 图像到: {save_path}")

        for ch in near_surface_channels + level500_channels:
            plot_one_channel(ch)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

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
            has_nan_inf_x = torch.isnan(x).any() or torch.isinf(x).any()
            has_nan_inf_y = torch.isnan(y).any() or torch.isinf(y).any()
            if has_nan_inf_x or has_nan_inf_y:
                if self.is_master:
                    times_str = ", ".join(str(t) for t in list(times))
                    print(f"[Train] batch {batch_idx} contains NaN/Inf, times: {times_str}")
                    print("[Train] 该 batch 已跳过，用于避免训练权重被 NaN 污染")
                continue

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

            # 如果 loss 本身出现 NaN/Inf，同样跳过该 batch，避免反向传播污染参数
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                if self.is_master:
                    times_str = ", ".join(str(t) for t in list(times))
                    print(f"[Train] batch {batch_idx} loss is NaN/Inf, times: {times_str}")
                    print("[Train] 该 batch 的梯度已跳过，请检查数据或数值稳定性")
                continue

            loss_item = float(loss.detach())
            recon_item = float(recon_loss.detach())
            kl_item = float(kl_loss.detach())

            total_loss += loss_item
            total_recon_loss += recon_item
            total_kl_loss += kl_item
            num_batches += 1

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
        if num_batches == 0:
            if self.is_master:
                print("[Train] 本 epoch 所有 batch 均因 NaN/Inf 被跳过，返回损失 0.0")
            avg_loss = 0.0
            avg_recon = 0.0
        else:
            avg_loss, avg_recon = self._all_reduce_loss(total_loss, total_recon_loss, num_batches)

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
