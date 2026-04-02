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
from models.gan import PatchDiscriminator
from trainers.fsdpgan import FSDPGANTrainer

import numpy as np
import pandas as pd
import multiprocessing as mp


try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def set_random_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
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
    times = np.array([pd.Timestamp(str(t)) for t in times])
    return torch.stack(x), torch.stack(y), torch.tensor(i), times


def main():
    if "RANK" not in os.environ:
        raise RuntimeError("mainfsdpgan.py 需要通过 torchrun 启动，例如: torchrun --nproc_per_node=2 maingan.py")

    device, rank, world_size = setup_distributed()
    is_master = rank == 0

    if is_master:
        print(f"World size = {world_size}, rank = {rank}, device = {device}")

    set_random_seed(42)
    data_sample_seed = 43

    # 与 baseline 一致：直接拟合 ERA5
    train_set = GFS2ERA5Dataset(
        start="2022-01-01 00:00:00",
        end="2024-12-31 18:00:00",
        x_path="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/gfs_2020_2024_c70_normalized",
        max_samples_per_year=490,
        sample_seed=data_sample_seed,
        target_mode="era5",
    )

    val_set = GFS2ERA5Dataset(
        start="2025-01-01 00:00:00",
        end="2025-11-20 18:00:00",
        x_path="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/gfs_2025_c70_normalized",
        val_sample_per_month=1,
        val_sample_year=2025,
        sample_seed=data_sample_seed,
        target_mode="era5",
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
    model = FSDP(base_model, device_id=device)

    discriminator = PatchDiscriminator(in_x_chans=70, in_y_chans=70, base_channels=64)

    num_epochs = 50
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=2e-5,
        betas=(0.9, 0.999),
    )

    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=2e-4,
        betas=(0.5, 0.999),
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=1e-4,
        min_lr=5e-7,
        verbose=(rank == 0),
    )

    trainer = FSDPGANTrainer(
        model=model,
        discriminator=discriminator,
        d_optimizer=d_optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=num_epochs,
        device=device,
        beta=1e-4,
        tb_dir="/home/ximutian/tensorboard_logs/swinunet_gan_2022_2024",
        save_dir="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/checkpoints/swinunet_gan_2022_2024",
        save_interval=1,
        use_amp=True,
        rank=rank,
        world_size=world_size,
        kl_anneal=False,
        kl_anneal_epochs=7,
        plot_root="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/channelpics/swinunet_gan_2022_2024",
        gan_start_epoch=35,
        l1_weight=10.0,
        adv_weight=1.0,
        fm_weight=5.0,
        d_grad_clip=5.0,
    )

    # 如需从已有 stage1 ckpt 继续，可设置 resume_path
    trainer.train(resume_path=None, only_model=False)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
