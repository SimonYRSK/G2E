import os
import random
import torch
from torch.utils.data import DataLoader, DistributedSampler
from trainers.fsdptrain import FSDPTrainer
from models.vae import G2E
from data import GFSReader, ERA5Reader, GFSERA5PairDataset, collate_fn

torch.backends.cudnn.deterministic = False   # 允许选择最优算法
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def set_random_seed(seed, rank):
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ["RANK"]) if is_distributed else 0
    world_size = int(os.environ["WORLD_SIZE"]) if is_distributed else 1
    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    print(f"using device: {device}")

    set_random_seed(42, rank)

    gfs_reader_train = GFSReader(start_dt="2020-01-01 00:00:00", end_dt="2024-12-31 18:00:00")
    era5_reader_train = ERA5Reader(start_dt="2020-01-01 00:00:00", end_dt="2024-12-31 18:00:00")
    gfs_reader_test = GFSReader(start_dt="2020-01-01 00:00:00", end_dt="2024-12-31 18:00:00")
    era5_reader_test = ERA5Reader(start_dt="2020-01-01 00:00:00", end_dt="2024-12-31 18:00:00")

    train_vars = [
        "Temperature",
        "2 metre temperature",
        "10 metre U wind component",
        "100 metre U wind component",
        "10 metre V wind component",
        "100 metre V wind component",
        "U component of wind",
        "V component of wind",
        "Geopotential height",
        "2 metre dewpoint temperature",
    ]

    train_dataset = GFSERA5PairDataset(
        gfs_reader=gfs_reader_train,
        era5_reader=era5_reader_train,
        gfs_vars=train_vars,
        normalize=True,
        norm_cache_path="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/gfs_2020_2024_c10/era5_norm_1_8.npz",
        base_layers=13,
        pad_mode="repeat",
    )
    test_dataset = GFSERA5PairDataset(
        gfs_reader=gfs_reader_test,
        era5_reader=era5_reader_test,
        gfs_vars=train_vars,
        normalize=True,
        norm_cache_path="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/gfs_2020_2024_c10/era5_norm_1_8.npz",
        base_layers=13,
        pad_mode="repeat",
    )

    if rank == 0:
        print("✅ 数据集初始化完成")

    batch_size = 1
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=42)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2, collate_fn=lambda x: collate_fn(x, base_layers=13)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=2, collate_fn=lambda x: collate_fn(x, base_layers=13)
    )

    if rank == 0:
        print(f"✅ DataLoader 加载完毕, 训练集: {len(train_dataset)}, 每卡: {len(train_dataset)//world_size}")

    model = G2E(
        in_ch=10,
        out_ch=10,
        widths=(32, 64, 128),
        encoder_cfg={
            "stage0": {"blocks": ["resblock"], "down": "conv"},
            "stage1": {"blocks": ["resblock"], "down": "conv"},
            "stage2": {"blocks": ["resblock"], "down": "conv"},
        },
        decoder_cfg={
            "stage0": {"blocks": ["resblock"], "up": "upsample"},
            "stage1": {"blocks": ["resblock"], "up": "upsample"},
        },
    )

    optimizer_config = {
        "type": "Adam",
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "betas": (0.9, 0.999),
    }
    scheduler_config = {
        "type": "CosineAnnealingLR",
        "T_max": 30,
        "eta_min": 1e-6,
    }

    trainer = FSDPTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=None,
        scheduler=None,
        epochs=3,
        device=device,
        beta=1e-4,                 # 稳定起步
        log_dir="./runs/experiment_fsdp",
        use_fsdp=True,
        save_dir="./checkpoints",
        save_interval=1,
        sharding_strategy="FULL_SHARD",
        mixed_precision=False,     # 先关混合精度，稳定后再开
        min_num_params=1e5,        # 确保子模块被分片
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        use_activation_ckpt=False, # 如显存仍高可改为 True
    )

    if rank == 0:
        print("✅ FSDPTrainer 初始化完成")

    trainer.train()

if __name__ == "__main__":
    main()
# 运行示例：
# torchrun --nproc_per_node=2 main.py