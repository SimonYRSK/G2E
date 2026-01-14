import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import random
from trainers.basetrain import BaseTrainer
from trainers.amptrain import AMPTrainer
from trainers.ddptrain import DDPTrainer
from trainers.fsdptrain import FSDPTrainer
from models.vae import G2E
from data import GFSReader, ERA5Reader, GFSERA5PairDataset, collate_fn
import os

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
    
    # ✅ 为训练集添加 DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        collate_fn=lambda x: collate_fn(x, base_layers=13)
    )

    # ✅ 为测试集添加 DistributedSampler
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=42
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
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
    
    # ✅✅✅ 在这里添加模型诊断（创建 Trainer 之前）
    if rank == 0:
        print("\n" + "="*80)
        print("🔍 模型诊断：检查 VAE 输出")
        print("="*80)
        
        # 将模型临时移到 device 进行测试
        model_test = model.to(device)
        model_test.eval()
        
        with torch.no_grad():
            # 创建小尺寸的测试输入（避免显存爆炸）
            dummy_input = torch.randn(1, 10, 721, 1440).to(device)  # 使用小尺寸
            
            try:
                x_recon, mu, log_var = model_test(dummy_input)
                
                print(f"✅ 前向传播成功")
                print(f"   输入形状: {dummy_input.shape}")
                print(f"   输出形状: {x_recon.shape}")
                print(f"   mu 形状: {mu.shape}")
                print(f"   log_var 形状: {log_var.shape}")
                print(f"\n📊 统计信息:")
                print(f"   mu 范围: [{mu.min():.4f}, {mu.max():.4f}]")
                print(f"   mu 均值: {mu.mean():.4f}")
                print(f"   log_var 范围: [{log_var.min():.4f}, {log_var.max():.4f}]")
                print(f"   log_var 均值: {log_var.mean():.4f}")
                
                # 计算 KL 散度
                kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                kl_per_element = kl / mu.numel()
                
                print(f"\n📈 KL 散度:")
                print(f"   总 KL: {kl.item():.4f}")
                print(f"   平均 KL (per element): {kl_per_element.item():.4f}")
                
                # 诊断
                if kl.item() > 10000:
                    print(f"\n⚠️  警告：KL 散度异常大 ({kl.item():.2f})！")
                    print(f"   可能原因：")
                    print(f"   1. log_var 初始化过大 (当前均值: {log_var.mean():.4f})")
                    print(f"   2. mu 初始化过大 (当前均值: {mu.mean():.4f})")
                    print(f"   3. beta 太小 (当前 beta: 1e-5)")
                    print(f"\n   建议：")
                    print(f"   1. 增大 beta 到 0.01 或 0.1")
                    print(f"   2. 检查 VAE 中 logvar 层的初始化")
                elif kl.item() < 0.01:
                    print(f"\n⚠️  警告：KL 散度过小 ({kl.item():.4f})，可能后验坍缩！")
                    print(f"   建议：减小 beta 或使用 KL annealing")
                else:
                    print(f"\n✅ KL 散度正常")
                
                # 检查重建损失
                recon = nn.functional.mse_loss(x_recon, dummy_input)
                print(f"\n📉 重建损失 (MSE): {recon.item():.4f}")
                
                # 检查输出是否有 NaN/Inf
                if torch.isnan(x_recon).any():
                    print(f"\n❌ 错误：输出包含 NaN！")
                if torch.isinf(x_recon).any():
                    print(f"\n❌ 错误：输出包含 Inf！")
                    
            except Exception as e:
                print(f"\n❌ 前向传播失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 将模型移回 CPU（为了 FSDP 包装）
        model = model_test.cpu()
        del model_test
        torch.cuda.empty_cache()
        
        print("="*80)
        print("🔍 模型诊断完成\n")
    
    # ✅ 优化器和调度器的配置
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
        optimizer=None,
        scheduler=None,
        epochs=3,
        device=device,
        beta=1e-5,  # ✅ 改为 0.01（从 1e-5）
        log_dir="./runs/experiment_fsdp",
        use_fsdp=True,
        save_dir="./checkpoints",
        save_interval=1,
        sharding_strategy="FULL_SHARD",
        mixed_precision=False,
        min_num_params=1e8,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
    )
    
    if rank == 0:
        print("✅ FSDPTrainer 初始化完成")
    
    trainer.train()


if __name__ == "__main__":
    main()
# export LD_PRELOAD=/home/ximutian/miniconda3/envs/xuyue/lib/libstdc++.so.6
# torchrun --nproc_per_node=2 main.py