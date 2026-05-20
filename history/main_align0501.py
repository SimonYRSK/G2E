import os
import random
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from data.pairset import GFS2ERA5Dataset
from models.swinUNET import G2E
from fuxi.fuxi_grad import UTransformer, FuXi
from fuxi_rmse_interface import FuXiRMSEInterface, DEFAULT_CHANNEL_WEIGHTS, TARGET_RMSE_CHANNELS
from trainers.fsdpalign_2phase import FSDPUNetAlignTrainer

try:
    from zarr.errors import ZarrUserWarning
except Exception:
    ZarrUserWarning = UserWarning


try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def configure_warning_filters():
    warnings.filterwarnings(
        "ignore",
        message=r"Both zarr\.json \(Zarr format 3\) and \.zgroup \(Zarr format 2\) metadata objects exist.*",
        category=ZarrUserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Object at .* is not recognized as a component of a Zarr hierarchy\.",
        category=ZarrUserWarning,
    )


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


def build_fuxi_model(device: torch.device, fuxi_dir: str) -> FuXi:
    conds = np.load(os.path.join(fuxi_dir, "conds.npy"))
    std = np.load(os.path.join(fuxi_dir, "std.npy"))
    mean = np.load(os.path.join(fuxi_dir, "mean.npy"))

    const = torch.from_numpy(conds).to(device=device, dtype=torch.float32)
    std_t = torch.from_numpy(std).to(device=device, dtype=torch.float32)
    mean_t = torch.from_numpy(mean).to(device=device, dtype=torch.float32)

    decoder = UTransformer(
        in_chans=75,
        out_chans=70,
        in_frames=2,
        image_size=(720, 1440),
        window_size=9,
        patch_size=4,
        down_times=1,
        embed_dim=1536,
        num_heads=24,
        depths=[12, 12, 12, 12],
    )

    model = FuXi(
        in_frames=2,
        out_frames=1,
        step_range=[1],
        decoder=[decoder, decoder, decoder],
        const=const,
        std=std_t,
        mean=mean_t,
        device=str(device),
        dtype=torch.float32,
    ).to(device=device, dtype=torch.float32)

    model.load(fuxi_dir, fmt="pth")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


def main():
    configure_warning_filters()

    if "RANK" not in os.environ:
        raise RuntimeError("main_align.py 需要通过 torchrun 启动，例如: torchrun --nproc_per_node=2 main_align0429.py")

    device, rank, world_size = setup_distributed()
    is_master = rank == 0

    if is_master:
        print(f"World size={world_size}, rank={rank}, device={device}")

    set_random_seed(42)

    x_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/gfs_2020_2025_c226_normalized"
    y_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanjiang/dataset/era5.2010_2025.c226.zarr"

    train_set = GFS2ERA5Dataset(
        start="2022-01-01 00:00:00",
        end="2024-12-31 18:00:00",
        x_path=x_path,
        y_path=y_path,
        max_samples_per_year=None,
        sample_seed=43,
    )

    val_set = GFS2ERA5Dataset(
        start="2025-01-01 00:00:00",
        end="2025-11-20 18:00:00",
        x_path=x_path,
        y_path=y_path,
        val_sample_per_month=4,
        val_sample_year=2025,
        sample_seed=43,
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
        dropout_rate=0.1,
        use_skip_connections=True,
        use_residual_blocks=True,
    )

    if is_master:
        print(f"G2E 参数量: {sum(p.numel() for p in base_model.parameters()) / 1e6:.2f} M")

    base_model.to(device)
    model = FSDP(base_model, device_id=device)

    fuxi_dir = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/fuxi_inference/main/fuxi"

    def build_fuxi_bundle():
        fuxi_model_local = build_fuxi_model(device, fuxi_dir=fuxi_dir)
        fuxi_rmse_local = FuXiRMSEInterface(
            fuxi_model=fuxi_model_local,
            era5_zarr_path=y_path,
            channel_names=train_set.target_channels,
            device=device,
            target_channels=TARGET_RMSE_CHANNELS,
            channel_weights=DEFAULT_CHANNEL_WEIGHTS,
            lead_hours=6,
        )
        return fuxi_model_local, fuxi_rmse_local

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=2e-5,
        betas=(0.9, 0.999),
    )

    num_epochs = 150
    warmup_epochs = 5
    min_lr = 5e-7

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(num_epochs - warmup_epochs, 1), eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    trainer = FSDPUNetAlignTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=num_epochs,
        device=device,
        beta=1e-4,
        tb_dir="/home/ximutian/tensorboard_logs/swinunet_align_0501",
        save_dir="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/checkpoints/swinunet_align_0501",
        save_interval=1,
        use_amp=False,
        rank=rank,
        world_size=world_size,
        kl_anneal=False,
        kl_anneal_epochs=7,
        plot_root="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/channelpics/swinunet_align_0501",
        recon_loss_type="l1",
        charbonnier_eps=1e-3,
        use_grad_loss=True,
        grad_loss_weight=0.4,
        l1_reg_weight=0.0,
        l2_reg_weight=0.0,
        fuxi_model=None,
        fuxi_rmse_interface=None,
        fuxi_loader=build_fuxi_bundle,
        channel_rmse_weight=1e-3,
        rmse_every_n_steps=1,
        rmse_samples_per_batch=1,
        pivot_epochs=30,
    )

    try:
        trainer.train(resume_path=None, only_model=False)
    finally:
        if getattr(trainer, "fuxi_rmse_interface", None) is not None:
            trainer.fuxi_rmse_interface.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
#export LD_LIBRARY_PATH=/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/conda_env/xmt/lib:$LD_LIBRARY_PATH