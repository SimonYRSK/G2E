import os
import random
import torch
from torch.utils.data import DataLoader, DistributedSampler
from trainers.train import BaseTrainer
from models.base import G2E
from data.data_ch import GFSReader, ERA5Reader, GFSERA5PairDataset, collate_fn

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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    set_random_seed(42)

    gfs_reader_train = GFSReader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2024-12-30 18:00:00",
    )
    era5_reader_train = ERA5Reader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2024-12-30 18:00:00",
    )
    # 验证：用一个一定存在的较早时间段（示例：2020-12 整月）
    gfs_reader_test = GFSReader(
        start_dt="2020-12-01 00:00:00",
        end_dt="2020-12-31 18:00:00",
    )
    era5_reader_test = ERA5Reader(
        start_dt="2020-12-01 00:00:00",
        end_dt="2020-12-31 18:00:00",
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
        "2 metre dewpoint temperature",
    ]

    train_dataset = GFSERA5PairDataset(
        gfs_reader=gfs_reader_train,
        era5_reader=era5_reader_train,
        gfs_vars=train_vars,
        normalize=True,
        norm_cache_path="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/gfs_2020_2024_c10/gfs_norm_2020_2024.npz",
        temporal_pair=False,
    )
    test_dataset = GFSERA5PairDataset(
        gfs_reader=gfs_reader_test,
        era5_reader=era5_reader_test,
        gfs_vars=train_vars,
        normalize=True,
        norm_cache_path="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/gfs_2020_2024_c10/gfs_norm_2020_2024.npz",
        temporal_pair=False,
    )

    print("✅ 数据集初始化完成")
    print(f"训练样本数: {len(train_dataset)}, 测试样本数: {len(test_dataset)}")
    print(f"展开后的总通道数 C: {train_dataset.total_channels}")
    print(f"通道名示例: {train_dataset.flat_var_names[:20]}")

    
    batch_size = 8
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=collate_fn,     # ✅ 新的 collate_fn，不再需要 base_layers
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print("✅ DataLoader 加载完毕")
    
    model = G2E(
        img_size=(721, 1440),
        patch_size=(4, 4),
        in_chans=58,
        embed_dim=1536,  
    ).to(device)

    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=30,
        eta_min=1e-6,
    )

    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=3,
        device=device,
        beta=1e-4,
        save_dir="/home/ximutian/checkpoints/test",
        save_interval=1,
        use_amp=False,   # 需要再提速可改为 True
    )


    print("✅ BaseTrainer 初始化完成（单卡训练）")
    trainer.train()


    trainer.train()

if __name__ == "__main__":
    main()
#export LD_LIBRARY_PATH=/home/ximutian/miniconda3/envs/xuyue/lib:$LD_LIBRARY_PATH