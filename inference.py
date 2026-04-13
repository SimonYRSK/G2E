import torch
from tqdm import tqdm
import os
from data.pairset import TARGET_CHANNELS, GFS2ERA5Dataset
from models.swinUNET import G2E

from torch.utils.data import DataLoader
import multiprocessing as mp
import numpy as np
import xarray as xr
import pandas as pd
import argparse

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
torch.backends.cudnn.deterministic = False   # 允许选择最优算法
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def inference(checkpoint_path, device, save_path, test_loader, gfs_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/gfs_2020_2024_c70_normalized"):
    print("load G2E")

    # 与训练 mainfsdp.py 中一致的 SwinUNet 配置
    # model = G2E(
    #     img_size=(721, 1440),
    #     patch_size=(4, 4),
    #     in_chans=70,
    #     out_chans=70,
    #     embed_dim=384,
    #     num_groups=32,
    #     num_heads=8,
    #     num_stages=3,
    #     window_size=9,
    #     depth=[0, 0, 1],
    #     using_checkpoints=True,
    #     using_time_embedding=True,
    #     res_per_stage=[1, 1, 1],
    #     channels=[384, 768, 1536],
    #     using_kl=False,
    # )

    # model = G2E(
    #     img_size=(721, 1440),
    #     patch_size=(2, 2),
    #     in_chans=70,
    #     out_chans=70,
    #     embed_dim=256,
    #     num_groups=32,
    #     num_heads=8,
    #     num_stages=3,
    #     window_size=9,
    #     depth=[1, 1, 2],
    #     using_checkpoints=True,
    #     using_time_embedding=True,
    #     res_per_stage=[1, 1, 1],
    #     channels=[256, 512, 1024],
    #     using_kl=False,
    # )


    model = G2E(
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
        use_skip_connections=False,
        use_residual_blocks=False,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    print("model loaded")

    model.eval()
    
    # ========== 关键调试：先检查DataLoader的样本数 ==========
    dataset_size = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    print(f"📊 数据集样本总数: {dataset_size}, 批次大小: {batch_size}, 预计批次数: {np.ceil(dataset_size / batch_size)}")
    
    if dataset_size == 0:
        raise ValueError("❌ 测试数据集为空！请检查GFS2ERA5Dataset的时间范围和数据路径是否正确")
    
    pbar = tqdm(test_loader, desc="推理进度")
    
    preds = []
    print("inferencing...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # 打印当前批次信息，确认循环在执行
            pbar.set_postfix({"批次": batch_idx + 1})
            
            # 兼容 dataset 返回 (x,y) 或 (x,y,i,times)
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                x, _, i, times = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, _ = batch
                i, times = None, None
            else:
                raise ValueError(f"Unexpected batch format: type={type(batch)}, len={len(batch) if hasattr(batch, '__len__') else 'N/A'}")

            x = x.to(device)
            # 当前 SwinUNet 仅使用时间特征，不再使用 i
            if times is not None:
                out = model(x, times=times)
            else:
                out = model(x)

            # using_kl=True 时，前向返回 (x_recon, mu, log_var)
            if isinstance(out, (tuple, list)):
                out = out[0]
            
            # ========== 检查输出是否有效 ==========
            if out is None:
                print(f"⚠️ 第 {batch_idx+1} 批次模型输出为空，跳过")
                continue
            
            # 将结果添加到列表
            pred_np = out.detach().cpu().numpy()
            preds.append(pred_np)
            print(f"✅ 第 {batch_idx+1} 批次推理完成，输出形状: {pred_np.shape}")
    
    # ========== 鲁棒性检查：确保preds非空 ==========
    if len(preds) == 0:
        raise ValueError("❌ 推理完成后preds列表为空！没有任何有效推理结果")
    
    # 拼接所有批次的结果
    arr = np.concatenate(preds, axis=0)
    print(f"📈 拼接后总结果形状: {arr.shape}")
    
    print("saving as zarr")
    # 与训练时一致：显式使用 consolidated=False，并按 TARGET_CHANNELS 选择/重排通道
    ds_gfs = xr.open_zarr(gfs_path, consolidated=False)

    time_list = test_loader.dataset.time_list
    # 先按时间再按通道名选择，确保 channel 顺序与 TARGET_CHANNELS 一致
    ds_gfs_sel = ds_gfs.sel(
        time=pd.to_datetime(time_list),
        channel=list(TARGET_CHANNELS),
    )

    new_ds = xr.Dataset(
        {
            "data": (("time", "channel", "lat", "lon"), arr)
        },
        coords={
            "time": ds_gfs_sel.time,
            "channel": ds_gfs_sel.channel,
            "lat": ds_gfs_sel.lat,
            "lon": ds_gfs_sel.lon,
        },
        attrs=ds_gfs_sel.attrs  # 可选，继承原有属性
    )

    assert list(ds_gfs_sel.channel.values) == list(TARGET_CHANNELS), "channel顺序不一致"
    assert arr.shape == ds_gfs_sel["data"].shape, f"shape不一致: {arr.shape} vs {ds_gfs_sel['data'].shape}"

    if os.path.exists(save_path):
        import shutil
        shutil.rmtree(save_path)
    new_ds.to_zarr(save_path, consolidated=True)
    print(f"✅ 推理结果已保存为zarr: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G2E inference runner")
    parser.add_argument("--start", type=str, default="2025-04-15 00:00:00")
    parser.add_argument("--end", type=str, default="2025-04-15 18:00:00")
    parser.add_argument("--x_path", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/gfs_2025_c70_normalized")
    parser.add_argument("--checkpoint_path", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/checkpoints/swinunet_2022_2024_3yr_3_27/checkpoint_epoch_23.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/inferenced/swinunet_2022_2024_3yr_3_27_20250415")
    parser.add_argument("--gfs_path", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/gfs_2025_c70_normalized")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=3)
    args = parser.parse_args()

    test_set = GFS2ERA5Dataset(
        start=args.start,
        end=args.end,
        x_path=args.x_path,
    )

    # ========== 调试：检查数据集是否加载成功 ==========
    print(f"🔍 测试数据集长度: {len(test_set)}")
    if len(test_set) == 0:
        print("❌ 警告：test_set 为空！请检查：")
        print(f"   1. 时间范围 {args.start} ~ {args.end} 是否在数据中")
        print(f"   2. x_path 路径是否正确：{args.x_path}")
        print("   3. GFS2ERA5Dataset 的时间解析逻辑是否正确")

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"loaded test set for {args.start} ~ {args.end}")

    inference(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        save_path=args.save_path,
        test_loader=test_loader,
        gfs_path=args.gfs_path,
    )
    #export LD_LIBRARY_PATH=/home/ximutian/miniconda3/envs/xuyue/lib:$LD_LIBRARY_PATH