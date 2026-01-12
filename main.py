import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import random
from trainers.basetrain import BaseTrainer
from trainers.amptrain import AMPTrainer
from models.vaebase import G2EVAE
from data import GFSReader, ERA5Reader, GFSERA5PairDataset, collate_fn


def set_random_seed(seed, rank):
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    gfs_reader_train = GFSReader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2024-12-31 18:00:00"
    )
    era5_reader_train = ERA5Reader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2024-12-31 18:00:00"
    )

    gfs_reader_test = GFSReader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2024-12-31 18:00:00"
    )
    era5_reader_test = ERA5Reader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2024-12-31 18:00:00"
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
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda x: collate_fn(x, base_layers=13)
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda x: collate_fn(x, base_layers=13)
    )
    print("dataloader加载完毕")

    model = G2EVAE(embed_dim=384, num_heads=6, window_size=7, depth = 4, latent_dim=384).to(device)


    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=1e-4,
        weight_decay=1e-5, 
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=30, 
        eta_min=1e-6
    )

    trainer = AMPTrainer(
        model = model,
        train_loader = train_dataloader,
        test_loader = test_dataloader,
        optimizer = optimizer,
        scheduler = scheduler,
        epochs = 20,
        device = device,
        beta = 1e-3
    )

    trainer.train()


if __name__ == "__main__":
    main()



# B = 1
# in_chans = out_chans = 10
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"using device: {device}")
# input = torch.randn(B, in_chans, 2, 721, 1440).to(device)
# print(f"input: {input.shape}")

# # # 测试UNet版本
# # fuxi_unet = FuxiUNet(embed_dim=864, num_heads=8, window_size=7).to(device)
# # output_unet = fuxi_unet(input)
# # print(f"\nUNet输出尺寸: {output_unet.shape}")

# # 测试VAE版本
# fuxi_vae = G2EVAE(embed_dim=864, num_heads=8, window_size=7, latent_dim=864).to(device)
# output_vae, mu, log_var = fuxi_vae(input)
# print(f"VAE输出尺寸: {output_vae.shape}")
# print(f"mu shape: {mu.shape}, log_var shape: {log_var.shape}")