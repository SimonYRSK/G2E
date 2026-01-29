import os
import random
import torch
from torch.utils.data import DataLoader, DistributedSampler
from trainers.train import BaseTrainer
from models.base import G2E
from data.pairset import GFS2ERA5Dataset

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

    train_set = GFS2ERA5Dataset(
        start="2023-01-01 00:00:00",
        end="2023-12-31 18:00:00",
    )

    dataloader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=False,  
        num_workers=3,  
        pin_memory=True, 
        drop_last=False,  
    )

    val_set = GFS2ERA5Dataset(
        start="2024-05-06 00:00:00",
        end="2024-05-07 18:00:00",
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=8,
        shuffle=False,  
        num_workers=3,  
        pin_memory=True, 
        drop_last=False,  
    )
    
    model = G2E(
        img_size=(721, 1440),
        patch_size=(4, 4),
        in_chans=70,
        embed_dim=1536, 
        depth = 8, 
    ).to(device)
    
    num_epochs = 70
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-5,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6,
    )

    trainer = BaseTrainer(
        model=model,
        train_loader=dataloader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=num_epochs,
        device=device,
        beta=5e-4,
        save_dir="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/checkpoints/baseline_1_29",
        save_interval=1,
        use_amp=False,   
    )



    trainer.train()
        #resume_path="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/checkpoints/baseline_1_28/checkpoint_epoch_14.pth",
        #only_model = True
    

if __name__ == "__main__":
    main()
#export LD_LIBRARY_PATH=/home/ximutian/miniconda3/envs/xuyue/lib:$LD_LIBRARY_PATH