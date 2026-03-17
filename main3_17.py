import os
import random
import torch
from torch.utils.data import DataLoader
from trainers.trainUNET import UNetTrainer
from models.swinUNET import G2E
from data.pairset import GFS2ERA5Dataset
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


def custom_collate(batch):
    x, y, i, times = zip(*batch)
    # times 保持为 pandas.Timestamp 数组，模型内部会转字符串再做时间特征
    times = np.array([pd.Timestamp(str(t)) for t in times])
    return torch.stack(x), torch.stack(y), torch.tensor(i), times


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    set_random_seed(42)

    train_set = GFS2ERA5Dataset(
        start="2023-10-01 00:00:00",
        end="2023-12-31 18:00:00",
    )

    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate,
    )

    val_set = GFS2ERA5Dataset(
        start="2024-03-15 00:00:00",
        end="2024-03-18 18:00:00",
    )

    val_loader = DataLoader(
        val_set,
        batch_size=8,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate,
    )

    model = G2E(
        img_size=(721, 1440),
        patch_size=(4, 4),
        in_chans=70,
        out_chans=70,
        embed_dim=1024,
        num_groups=32,
        num_heads=8,
        num_stages=3,
        window_size=9,
        depth=6,
        using_checkpoints=True,
        using_time_embedding=True,
        res_per_stage=[1, 2, 4],
    ).to(device)

    num_epochs = 200

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-6,
        weight_decay=1e-5,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=False,
        min_lr=5e-7,
    )

    trainer = UNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=num_epochs,
        device=device,
        beta=0.0,
        tb_dir="./tensorboard_logs/unet3_17",
        save_dir="./checkpoints/unet3_17",
        save_interval=1,
        use_amp=False,
    )

    trainer.train(
        resume_path=None,
        only_model=False,
    )


if __name__ == "__main__":
    main()
