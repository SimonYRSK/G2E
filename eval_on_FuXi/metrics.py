import os
import numpy as np
import xarray as xr
import pandas as pd
import zarr
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

# 新路径
PRED_ROOT_RTM = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/huangqiusheng/eval/RTM_base_6h/20240101-12"
PRED_ROOT_ERA5 = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/era5/20240101-12"
PRED_ROOT_NAIVE_GFS = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/inference_naive_gfs/20240101-12"
PRED_ROOT_TRANS_GFS = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/inference_trans_gfs/20240101-12"
ERA5_ROOT = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/huangqiusheng/datasets/era5.rtm.02_25.6h.c109.new3/"
TARGET_CHANNEL = "z500"

# 第一步：确定时间步范围（STEPS）
def get_steps(pred_root):
    files = os.listdir(pred_root)
    zarr_files = [f for f in files if f.endswith('.zarr')]
    steps = [f.split('.')[0] for f in zarr_files if f.split('.')[0].isdigit()]
    steps = sorted(steps)  # 按字符串排序，如 '001', '002', ...
    if not steps:
        raise ValueError(f"No .zarr files found in {pred_root}")
    print(f"Found steps: {steps}")
    return steps

# 获取预测 z500
def get_pred_z500(pred_dir, time):
    ds = xr.open_zarr(pred_dir)
    chs = ds['channel'].values
    if chs.dtype == 'object':
        chs = [ch.decode() if isinstance(ch, bytes) else str(ch) for ch in chs]
    chs = np.array(chs)
    idx_arr = np.where(chs == TARGET_CHANNEL)[0]
    if len(idx_arr) == 0:
        raise ValueError(f"{TARGET_CHANNEL} not found in {pred_dir}")
    idx = idx_arr[0]
    arr = ds['output'][0,0,0,idx,:,:]
    # 添加 time 坐标
    arr = arr.expand_dims(time=[pd.Timestamp(time)])
    ds.close()
    return arr

# 获取真值 z500
def get_true_z500(time):
    ds = xr.open_zarr(ERA5_ROOT)
    chs = ds['channel'].values
    if chs.dtype == 'object':
        chs = [ch.decode() if isinstance(ch, bytes) else str(ch) for ch in chs]
    chs = np.array(chs)
    idx_arr = np.where(chs == TARGET_CHANNEL)[0]
    if len(idx_arr) == 0:
        raise ValueError(f"{TARGET_CHANNEL} not found in ERA5")
    idx = idx_arr[0]
    t_idx = np.where(pd.to_datetime(ds['time'].values) == pd.Timestamp(time))[0][0]
    arr = ds['data'][t_idx, idx, :, :]
    # 反归一化
    m = xr.open_dataarray(os.path.join(ERA5_ROOT, 'mean.nc')).values
    s = xr.open_dataarray(os.path.join(ERA5_ROOT, 'std.nc')).values
    if m.ndim == 1:
        m = m[:, np.newaxis, np.newaxis]
        s = s[:, np.newaxis, np.newaxis]
    arr = arr * s[idx] + m[idx]
    # 转为DataArray并加上lat/lon/time坐标
    arr = xr.DataArray(arr, dims=('lat', 'lon'),
                       coords={'lat': ds['lat'].values, 'lon': ds['lon'].values})
    arr = arr.expand_dims(time=[pd.Timestamp(time)])
    ds.close()
    return arr

# 计算 RMSE
def calc_rmse(pred, true):
    # pred, true: xarray.DataArray，带lat/lon
    weights = np.cos(np.deg2rad(np.abs(true.lat)))
    error = (pred - true) ** 2
    rmse = np.sqrt(error.weighted(weights).mean(("lat", "lon")))
    return float(rmse.compute())

def get_anomaly(x, clim):
    # x: DataArray, 带有 time 维
    cmean = clim['z500'].sel(doy=x.time.dt.dayofyear, hour=x.time.dt.hour)
    return x - cmean

def compute_acc(out, tgt, clim):
    out = get_anomaly(out, clim)
    tgt = get_anomaly(tgt, clim)
    wlat = np.cos(np.deg2rad(tgt.lat))
    wlat /= wlat.mean()
    A = (wlat * out * tgt).sum(("lat", "lon"), skipna=True)
    B = (wlat * out**2).sum(("lat", "lon"), skipna=True)
    C = (wlat * tgt**2).sum(("lat", "lon"), skipna=True)
    acc = A / np.sqrt(B * C + 1e-12)
    return float(acc.compute())  # 先compute，再转float

# 主逻辑
STEPS = get_steps(PRED_ROOT_TRANS_GFS)  # 自动确定 STEPS

# 预测起始时间（需与预测步长对应）
start_time = pd.Timestamp("2024-01-01 12:00:00")  # 原代码中的起始时间，假设相同
hour_interval = 6  # 步长间隔
times = [start_time + pd.Timedelta(hours=hour_interval * int(step)) for step in STEPS]  # 注意 int(step) 因为 step 是 '001' 等
clim = xr.open_zarr("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanjiang/eval/era5/clim.daily")

rmse_rtm = []
rmse_era5 = []
rmse_naive_gfs = []
rmse_trans_gfs = []

acc_rtm = []
acc_era5 = []
acc_naive_gfs = []
acc_trans_gfs = []

for i, step in enumerate(STEPS):
    trans_gfs_dir = os.path.join(PRED_ROOT_TRANS_GFS, f"{step}.zarr")
    naive_gfs_dir = os.path.join(PRED_ROOT_NAIVE_GFS, f"{step}.zarr")
    era5_dir = os.path.join(PRED_ROOT_ERA5, f"{step}.zarr")
    # 预测
    pred_trans_gfs = get_pred_z500(trans_gfs_dir, times[i])
    pred_naive_gfs = get_pred_z500(naive_gfs_dir, times[i])
    pred_era5 = get_pred_z500(era5_dir, times[i])
    # 真实
    true = get_true_z500(times[i])
    # 评估 RMSE
    rmse_trans_gfs.append(calc_rmse(pred_trans_gfs, true))
    rmse_naive_gfs.append(calc_rmse(pred_naive_gfs, true))
    rmse_era5.append(calc_rmse(pred_era5, true))

    acc_trans_gfs.append(compute_acc(pred_trans_gfs, true, clim))
    acc_naive_gfs.append(compute_acc(pred_naive_gfs, true, clim))
    acc_era5.append(compute_acc(pred_era5, true, clim))
    #print(f"Step {step}:ERA5 RMSE={rmse_era5[-1]:.3f}, ACC={acc_era5[-1]:.3f} | Trans GFS RMSE={rmse_trans_gfs[-1]:.3f}, ACC={acc_trans_gfs[-1]:.3f}")

    print(f"Step {step}:Naive ERA5 RMSE={rmse_era5[-1]:.3f}, ACC={acc_era5[-1]:.3f} | Naive GFS RMSE={rmse_naive_gfs[-1]:.3f}, ACC={acc_naive_gfs[-1]:.3f} | GFS2ERA5 RMSE={rmse_trans_gfs[-1]:.3f}, ACC={acc_trans_gfs[-1]:.3f}")


plt.figure(figsize=(10,5))
plt.plot([int(s) for s in STEPS], acc_era5, label='Naive_ERA5 ACC', marker='o')
plt.plot([int(s) for s in STEPS], acc_naive_gfs, label='Naive_GFS ACC', marker='o')
plt.plot([int(s) for s in STEPS], acc_trans_gfs, label='GFS_2_ERA5 ACC', marker='o')
plt.xlabel('Forecast Step')
plt.ylabel('ACC')
plt.title('ACC (z500)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("z500_acc_rtm_curve.png")
plt.close()

plt.figure(figsize=(10,5))
plt.plot([int(s) for s in STEPS], rmse_era5, label='Naive_ERA5 RMSE', marker='o')
plt.plot([int(s) for s in STEPS], rmse_naive_gfs, label='Naive_GFS RMSE', marker='o')
plt.plot([int(s) for s in STEPS], rmse_trans_gfs, label='GFS_2_ERA5 RMSE', marker='o')
plt.xlabel('Forecast Step')
plt.ylabel('RMSE')
plt.title('RMSE (z500)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("z500_rmse_rtm_curve.png")
plt.close()


print("评估完成，曲线已保存为 z500_acc_rtm_curve.png")