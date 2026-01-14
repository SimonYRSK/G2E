import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler  # ✅ 导入
import random
from trainers.basetrain import BaseTrainer
from trainers.amptrain import AMPTrainer
from trainers.ddptrain import DDPTrainer
from trainers.fsdptrain import FSDPTrainer
from models.vae import G2E
from data import GFSReader, ERA5Reader, GFSERA5PairDataset, collate_fn
import os  # ✅ 导入

def set_random_seed(seed, rank):
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    # ✅ 检查是否在 DDP 模式
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        world_size = 1

    # ✅ 设置随机种子
    set_random_seed(42, rank)

    gfs_reader_train = GFSReader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2020-01-02 18:00:00" #"2024-12-31 18:00:00"
    )
    era5_reader_train = ERA5Reader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2020-01-02 18:00:00"
    )

    gfs_reader_test = GFSReader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2020-01-02 18:00:00"
    )
    era5_reader_test = ERA5Reader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2020-01-02 18:00:00"
    )

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
        "2 metre dewpoint temperature"
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

    print(f"✅ 数据集初始化完成")

    batch_size = 1
    
    # ✅ 为训练集添加 DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,  # ✅ 每轮随机打乱
        seed=42
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # ✅ 用 sampler 替代 shuffle
        num_workers=2,
        collate_fn=lambda x: collate_fn(x, base_layers=13)
    )

    # ✅ 为测试集添加 DistributedSampler（可选，但推荐）
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,  # 测试集不需要随机
        seed=42
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,  # ✅ 用 sampler
        num_workers=2,
        collate_fn=lambda x: collate_fn(x, base_layers=13)
    )
    
    if rank == 0:
        print(f"✅ DataLoader 加载完毕")
        print(f"   训练集样本总数: {len(train_dataset)}")
        print(f"   每张 GPU 分配: {len(train_dataset) // world_size}")

    #=========================================================================================================================================================
    #=========================================================================================================================================================
    
    model = G2E(
        in_ch=10, out_ch=10,
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

        
    # ✅ 优化器和调度器的配置（传给 FSDPTrainer）
    optimizer_config = {
        'type': 'Adam',
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'betas': (0.9, 0.999),
    }

    scheduler_config = {
        'type': 'CosineAnnealingLR',
        'T_max': 30,
        'eta_min': 1e-6,
    }

    trainer = FSDPTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=None,  # ✅ 传 None，由 Trainer 内部创建
        scheduler=None,  # ✅ 传 None
        epochs=3,
        device=device,
        beta=1e-5,
        log_dir="./runs/experiment_fsdp",
        use_fsdp=True,
        save_dir="./checkpoints",
        save_interval=1,
        sharding_strategy="FULL_SHARD",
        mixed_precision=False,
        min_num_params=1e6,
        optimizer_config=optimizer_config,  # ✅ 传配置
        scheduler_config=scheduler_config,  # ✅ 传配置
    )

# ...existing code...
    
    if rank == 0:
        print("✅ DDPTrainer 初始化完成")
    
    trainer.train()


if __name__ == "__main__":
    main()
# export LD_PRELOAD=/home/ximutian/miniconda3/envs/xuyue/lib/libstdc++.so.6
# torchrun --nproc_per_node=2 main.py