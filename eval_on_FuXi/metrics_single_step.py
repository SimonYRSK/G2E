import os
import numpy as np
import xarray as xr
import pandas as pd
import zarr
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ===================== 核心配置 =====================
PRED_ROOT_ERA5 = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/era5/20220101-12/001.zarr"
PRED_ROOT_NAIVE_GFS = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/naive_gfs/20220101-12/001.zarr"
ROOT = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/huangqiusheng/datasets/era5.rtm.02_25.6h.c109.new3/"

TARGET_CHANNELS = [
    "t200", "u200", "v200", "z200", "q200",
    "msl", "tp", "t2m", "u10m", "v10m"
]

ERA5_TARGET_TIME = pd.Timestamp("2022-01-01 12:00:00")

SAVE_FIG_DIR = "./eval_figures"  # 保存图片的目录

# ===================== 工具函数 =====================
def read_zarr_array_light(path, slice_idx=None):
    """轻量读取 zarr 数组"""
    zarr_arr = zarr.open(path, mode='r')
    if slice_idx is not None:
        return np.array(zarr_arr[slice_idx])
    return zarr_arr.shape, zarr_arr.dtype


def read_pred_data_light(pred_root):
    """读取预测 zarr 的 output 数据（取第一个成员、第一个步）"""
    print(f"读取预测数据: {os.path.basename(pred_root)}")
    
    channel_path = os.path.join(pred_root, "channel")
    zarr_channel = zarr.open(channel_path, mode='r')
    pred_channel = np.array(zarr_channel[:])
    if pred_channel.dtype == 'object':
        pred_channel = [ch.decode() if isinstance(ch, bytes) else str(ch) for ch in pred_channel]

    output_path = os.path.join(pred_root, "output")
    zarr_output = zarr.open(output_path, mode='r')
    
    # 假设 shape = (1,1,1,channel,lat,lon) 或类似，取第一个
    pred_data = np.array(zarr_output[0, 0, 0, :, :, :])  # channel, lat, lon
    
    print(f"  shape: {pred_data.shape}, 通道数: {len(pred_channel)}")
    return pred_channel, pred_data


def read_era5_true_data(era5_root, target_time):
    """读取真实 ERA5 数据（目标时间步，已反归一化）"""
    print("读取真实 ERA5 数据...")
    ds_era5 = xr.open_zarr(era5_root, chunks={"time": 1})
    
    era5_time = pd.to_datetime(ds_era5.time.values)
    time_mask = era5_time == target_time
    if not np.any(time_mask):
        raise ValueError(f"未在 ERA5 中找到时间 {target_time}")
    
    time_idx = np.where(time_mask)[0][0]
    print(f"  目标时间索引: {time_idx}")
    
    era5_data = ds_era5['data'].isel(time=time_idx).compute().values  # channel, lat, lon
    
    # 反归一化
    m = xr.open_dataarray(os.path.join(era5_root, 'mean.nc')).values
    s = xr.open_dataarray(os.path.join(era5_root, 'std.nc')).values
    if m.ndim == 1:
        m = m[:, np.newaxis, np.newaxis]
        s = s[:, np.newaxis, np.newaxis]
    era5_data = era5_data * s + m
    
    era5_channel = np.array(ds_era5['channel'][:])
    if era5_channel.dtype == 'object':
        era5_channel = [ch.decode() if isinstance(ch, bytes) else str(ch) for ch in era5_channel]
    
    print(f"  ERA5 shape: {era5_data.shape}")
    return era5_channel, era5_data


def calculate_metrics(pred, true):
    """计算 MAE, RMSE, Bias（忽略 NaN）"""
    mask = ~np.isnan(true) & ~np.isnan(pred)
    if np.sum(mask) == 0:
        return np.nan, np.nan, np.nan
    p = pred[mask]
    t = true[mask]
    mae = np.mean(np.abs(p - t))
    rmse = np.sqrt(np.mean((p - t)**2))
    bias = np.mean(p - t)
    return mae, rmse, bias


def plot_comparison(true, pred_era5, pred_gfs, channel_name, lon_idx=720, save_path=None):
    """
    绘制沿纬度剖面对比图（固定经度）
    lon_idx: 经度索引（默认中间 720 ≈ 0°）
    """
    lats = np.linspace(90, -90, 721)  # 假设 ERA5 lat 从90到-90
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(lats, true[:, lon_idx], label='ERA5 True', color='black', linewidth=2.5)
    ax.plot(lats, pred_era5[:, lon_idx], label='FuXi-ERA5 Init', color='blue', linestyle='--')
    ax.plot(lats, pred_gfs[:, lon_idx], label='FuXi-GFS Init', color='red', linestyle='-.')
    
    ax.set_xlabel('Latitude (°N)')
    ax.set_ylabel(f'{channel_name} value')
    ax.set_title(f'{channel_name} - Latitude Profile at lon≈0°\n2022-01-01 12:00')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  保存图片: {save_path}")
    plt.close()


def plot_overall_bar_comparison(metrics_dict, save_path=None):
    """
    绘制所有通道的 MAE 和 RMSE 对比柱状图
    - 使用子图，每行 5 个变量（假设 10 个通道，2行5列）
    - 每个子图：x = ['MAE', 'RMSE']，每个指标下 ERA5 和 GFS 半透明柱子并排（轻微重叠）
    - 动态调整每个子图 y 轴范围
    """
    num_vars = len(metrics_dict)
    num_cols = 5
    num_rows = (num_vars + num_cols - 1) // num_cols  # 计算行数
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4), squeeze=False)
    fig.suptitle('ERA5 vs GFS Initialization: MAE & RMSE Comparison', fontsize=16, y=1.02)
    
    # 颜色定义：ERA5 蓝半透明，GFS 红半透明
    color_era5 = 'blue'
    color_gfs = 'red'
    alpha = 0.6  # 半透明度
    
    for i, (ch, metrics) in enumerate(metrics_dict.items()):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        
        # 数据：['MAE', 'RMSE']
        indicators = ['MAE', 'RMSE']
        x = np.arange(len(indicators))  # 0: MAE, 1: RMSE
        
        # ERA5 和 GFS 值
        era5_vals = [metrics['MAE_ERA5'], metrics['RMSE_ERA5']]
        gfs_vals = [metrics['MAE_GFS'], metrics['RMSE_GFS']]
        
        # 柱宽（轻微重叠）
        width = 0.4
        
        # 绘制 ERA5 柱（左移一点）
        ax.bar(x - width/2, era5_vals, width, label='ERA5 Init', color=color_era5, alpha=alpha)
        
        # 绘制 GFS 柱（右移一点，重合部分）
        ax.bar(x + width/2, gfs_vals, width, label='GFS Init', color=color_gfs, alpha=alpha)
        
        ax.set_xticks(x)
        ax.set_xticklabels(indicators)
        ax.set_title(ch)
        ax.set_ylabel('Value')
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 动态 y 轴范围（基于 max 值 + 10% 裕度）
        max_val = max(max(era5_vals), max(gfs_vals)) * 1.1
        ax.set_ylim(0, max_val if max_val > 0 else 1)
    
    # 隐藏多余子图
    for j in range(i + 1, num_rows * num_cols):
        row = j // num_cols
        col = j % num_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  保存整体对比柱状图: {save_path}")
    plt.close()


# ===================== 主流程 =====================
if __name__ == "__main__":
    os.makedirs(SAVE_FIG_DIR, exist_ok=True)
    
    try:
        # 1. 读取两个预测结果
        print("="*60)
        print("读取 FuXi-ERA5 初始化预测...")
        era5_init_ch, era5_init_data = read_pred_data_light(PRED_ROOT_ERA5)
        
        print("\n读取 FuXi-GFS 初始化预测...")
        gfs_init_ch, gfs_init_data = read_pred_data_light(PRED_ROOT_NAIVE_GFS)
        
        # 2. 读取真实 ERA5
        true_ch, true_data = read_era5_true_data(ROOT, ERA5_TARGET_TIME)
        
        # 3. 通道匹配字典
        era5_init_dict = {ch: i for i, ch in enumerate(era5_init_ch)}
        gfs_init_dict  = {ch: i for i, ch in enumerate(gfs_init_ch)}
        true_dict      = {ch: i for i, ch in enumerate(true_ch)}
        
        # 4. 计算并打印误差 + 绘图
        print("\n" + "="*70)
        print(f"{'通道':<8} {'MAE (ERA5 init)':<15} {'RMSE':<12} {'Bias':<10} | "
              f"{'MAE (GFS init)':<15} {'RMSE':<12} {'Bias':<10}")
        print("-"*70)
        
        # 收集指标，用于整体柱状图
        metrics_dict = {}
        
        for ch in TARGET_CHANNELS:
            if ch not in true_dict:
                print(f"{ch:<8} {'缺失真实值':<60}")
                continue
            
            true_arr = true_data[true_dict[ch]]
            
            # FuXi-ERA5 初始化
            if ch in era5_init_dict:
                pred_era5 = era5_init_data[era5_init_dict[ch]]
                mae_e, rmse_e, bias_e = calculate_metrics(pred_era5, true_arr)
            else:
                mae_e = rmse_e = bias_e = np.nan
            
            # FuXi-GFS 初始化
            if ch in gfs_init_dict:
                pred_gfs = gfs_init_data[gfs_init_dict[ch]]
                mae_g, rmse_g, bias_g = calculate_metrics(pred_gfs, true_arr)
            else:
                mae_g = rmse_g = bias_g = np.nan
            
            print(f"{ch:<8} {mae_e:>12.4f} {rmse_e:>12.4f} {bias_e:>10.4f} | "
                  f"{mae_g:>12.4f} {rmse_g:>12.4f} {bias_g:>10.4f}")
            
            # 收集指标
            metrics_dict[ch] = {
                'MAE_ERA5': mae_e,
                'RMSE_ERA5': rmse_e,
                'MAE_GFS': mae_g,
                'RMSE_GFS': rmse_g
            }
            
            # 绘图对比（沿纬度剖面）
            if ch in era5_init_dict and ch in gfs_init_dict:
                fig_path = os.path.join(SAVE_FIG_DIR, f"{ch}_comparison_20220101-12.png")
                plot_comparison(
                    true_arr,
                    pred_era5,
                    pred_gfs,
                    ch,
                    lon_idx=720,          # ≈0° 经度
                    save_path=fig_path
                )
        
        print("="*70)
        print(f"所有计算完成！图片保存在: {SAVE_FIG_DIR}")
        
        # 5. 绘制整体对比柱状图
        overall_fig_path = os.path.join(SAVE_FIG_DIR, "overall_mae_rmse_comparison.png")
        plot_overall_bar_comparison(metrics_dict, save_path=overall_fig_path)
        
    except Exception as e:
        print(f"\n❌ 执行出错: {str(e)}")
        import traceback
        traceback.print_exc()