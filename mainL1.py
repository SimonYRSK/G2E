import os
import random

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from data.pairset import GFS2ERA5Dataset
from models.swinUNET import G2E
from trainers.fsdptrain import FSDPUNetTrainer
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import numpy as np
import pandas as pd
import multiprocessing as mp


try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


torch.backends.cudnn.deterministic = False   # 允许选择最优算法
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def set_random_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def setup_distributed():
    """初始化单机多卡分布式环境，返回 device, rank, world_size。"""
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return device, rank, world_size


def custom_collate(batch):
    x, y, i, times = zip(*batch)
    # times 保持为 pandas.Timestamp 数组，模型内部会转字符串再做时间特征
    times = np.array([pd.Timestamp(str(t)) for t in times])
    return torch.stack(x), torch.stack(y), torch.tensor(i), times


def main():
    if "RANK" not in os.environ:
        raise RuntimeError("mainfsdp.py 需要通过 torchrun 启动，例如: torchrun --nproc_per_node=2 mainfsdp.py")

    device, rank, world_size = setup_distributed()
    is_master = (rank == 0)

    if is_master:
        print(f"World size = {world_size}, rank = {rank}, device = {device}")

    set_random_seed(42)

    data_sample_seed = 43

    # 重建损失配置：
    # - "l2"    : 仅 MSE（默认）
    # - "l1"    : 仅 L1
    recon_loss_type = "l1"

    # 1) 训练集：使用 2022-2024，全量样本，来自 2020-2024 标准化 GFS Zarr
    train_set = GFS2ERA5Dataset(
        start="2022-01-01 00:00:00",
        end="2024-12-31 18:00:00",
        x_path="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/gfs_2020_2024_c70_normalized",
        # max_samples_per_year 可在调参时设成一个较小的数，例如 500 或 1000，快速训练
        # 正式训练时设为 None 即可使用全量数据
        max_samples_per_year=490,
        sample_seed = data_sample_seed,
    )

    # 2) 验证集：使用 2025 年数据，按原逻辑在 2025 年每个月随机抽取若干“整天”的所有时间步
    val_set = GFS2ERA5Dataset(
        start="2025-01-01 00:00:00",
        end="2025-11-20 18:00:00",
        x_path="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/gfs_2025_c70_normalized",
        val_sample_per_month=1,
        val_sample_year=2025,
        sample_seed = data_sample_seed,
    )

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_set,
        batch_size=4,
        shuffle=False,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate,
        prefetch_factor=1,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=4,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate,
        prefetch_factor=1,
    )
    # 先在未包裹 FSDP 的模型上统计参数量（只在 rank0）
    base_model = G2E(
        img_size=(721, 1440),
        patch_size=(4, 4),
        in_chans=70,
        out_chans=70,
        embed_dim=384,
        num_groups=32,
        num_heads=8,
        num_stages=3,
        window_size=9,
        depth=[0, 0, 1],
        using_checkpoints=True,
        using_time_embedding=True,
        res_per_stage=[1, 1, 1],
        channels=[384, 768, 1536],
        using_kl=False,
    )

    if is_master:
        print(f"模型参数量: {sum(p.numel() for p in base_model.parameters()) / 1e6:.2f} M")

    base_model.to(device)

    # 用 FSDP 包裹模型
    model = FSDP(base_model, device_id=device)

    num_epochs = 120

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=2e-5,
        betas=(0.9, 0.999),
    )

    min_lr = 5e-7

    # 使用 warmup + 余弦退火学习率调度器（按 epoch 进行 step）
    warmup_epochs = 5
    # 线性 warmup：从 0.1×lr 线性增加到 1.0×lr
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    # 余弦退火：从当前 lr 逐步衰减到 min_lr
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(num_epochs - warmup_epochs, 1),
        eta_min=min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    trainer = FSDPUNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=num_epochs,
        device=device,
        beta=1e-4,  # KL 目标权重，如未使用 KL 可设为 0
        tb_dir="/home/ximutian/tensorboard_logs/swinunetL1_2022_2024_1yr_4_2",
        save_dir="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/checkpoints/swinunetL1_2022_2024_1yr_4_2",
        save_interval=1,
        use_amp=True,
        rank=rank,
        world_size=world_size,
        kl_anneal=False,           # 启用 KL annealing
        kl_anneal_epochs=7,      # 前 10 个 epoch 从 0 线性涨到 beta
        plot_root="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/channelpics/swinunetL1_2022_2024_1yr_4_2",
        recon_loss_type=recon_loss_type
    )

    trainer.train(
        resume_path=None,
        only_model=False,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
#export LD_LIBRARY_PATH=/home/ximutian/miniconda3/envs/xuyue/lib:$LD_LIBRARY_PATH