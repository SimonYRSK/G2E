import os
import random
import torch
from torch.utils.data import DataLoader, DistributedSampler
from trainers.train import BaseTrainer
from models.base import G2E
from data import GFS2ERA5Dataset

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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    set_random_seed(42)

    train_set = GFS2ERA5Dataset()

    dataloader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=False,  # 训练时设为True，测试耗时设为False避免额外开销
        num_workers=6,  # 先测试单线程，后续可改为多线程对比
        pin_memory=True,  # GPU训练时开启，加速数据传输
        drop_last=False,  # 保留最后一个不足batch_sizes的批次
    )
    
    
    
    model = G2E(
        img_size=(721, 1440),
        patch_size=(4, 4),
        in_chans=70,
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
        train_loader=dataloader,
        test_loader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=3,
        device=device,
        beta=1e-4,
        save_dir="/home/ximutian/checkpoints/test",
        save_interval=1,
        use_amp=False,   
    )



    trainer.train()

if __name__ == "__main__":
    main()
#export LD_LIBRARY_PATH=/home/ximutian/miniconda3/envs/xuyue/lib:$LD_LIBRARY_PATH