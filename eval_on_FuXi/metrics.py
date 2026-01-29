import os
import numpy as np
import xarray as xr
import pandas as pd
import zarr
import matplotlib.pyplot as plt
import warnings
from pathlib import Path




PRED_ROOT_ERA5 = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/era5/20221201-12"
PRED_ROOT_TRANS_GFS = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/inference_trans_gfs/20221201-12"
ERA5_ROOT = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/huangqiusheng/datasets/era5.rtm.02_25.6h.c109.new3/"
TARGET_CHANNEL = "z500"
STEPS = [f"{i:03d}" for i in range(1, 39)]  # 001~038

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

def calc_rmse(pred, true):
    mask = ~np.isnan(pred) & ~np.isnan(true)
    return np.sqrt(np.mean((pred[mask] - true[mask]) ** 2))

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
    return acc.item()  # 转为标量

# 预测起始时间（需与预测步长对应）
start_time = pd.Timestamp("2022-12-10 12:00:00")
hour_interval = 6  # 步长间隔
times = [start_time + pd.Timedelta(hours=hour_interval * i) for i in range(len(STEPS))]
clim = xr.open_zarr("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanjiang/eval/era5/clim.daily")

ds_era5 = xr.open_zarr(ERA5_ROOT)
lat_arr = ds_era5['lat'].values
ds_era5.close()

rmse_era5, acc_era5 = [], []
rmse_gfs, acc_gfs = [], []
mean_era5, mean_pred_era5, mean_pred_gfs = [], [], []
for i, step in enumerate(STEPS):
    pred_dir_era5 = os.path.join(PRED_ROOT_ERA5, f"{step}.zarr")
    pred_dir_gfs = os.path.join(PRED_ROOT_TRANS_GFS, f"{step}.zarr")
    # 预测
    pred_era5 = get_pred_z500(pred_dir_era5, times[i])
    pred_gfs = get_pred_z500(pred_dir_gfs, times[i])
    # 真实
    true = get_true_z500(times[i])
    # 评估
    rmse_era5.append(calc_rmse(pred_era5.values, true.values))
    rmse_gfs.append(calc_rmse(pred_gfs.values, true.values))
    acc_era5.append(compute_acc(pred_era5, true, clim))
    acc_gfs.append(compute_acc(pred_gfs, true, clim))
    mean_era5.append(np.nanmean(true.values))
    mean_pred_era5.append(np.nanmean(pred_era5.values))
    mean_pred_gfs.append(np.nanmean(pred_gfs.values))
    print(f"Step {step}: ERA5 RMSE={rmse_era5[-1]:.3f}, ACC={acc_era5[-1]:.3f} | GFS RMSE={rmse_gfs[-1]:.3f}, ACC={acc_gfs[-1]:.3f}")
# 绘图
plt.figure(figsize=(10,5))
plt.plot(range(1,39), rmse_era5, label='FuXi-ERA5 Init RMSE', marker='o')
plt.plot(range(1,39), rmse_gfs, label='FuXi-GFS Init RMSE', marker='o')
plt.xlabel('Forecast Step')
plt.ylabel('RMSE')
plt.title('RMSE vs Forecast Step (z500)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("z500_rmse_curve.png")
plt.close()

plt.figure(figsize=(10,5))
plt.plot(range(1,39), acc_era5, label='FuXi-ERA5 Init ACC', marker='o')
plt.plot(range(1,39), acc_gfs, label='FuXi-GFS Init ACC', marker='o')
plt.xlabel('Forecast Step')
plt.ylabel('ACC')
plt.title('ACC vs Forecast Step (z500)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("z500_acc_curve.png")
plt.close()


plt.figure(figsize=(10,5))
plt.plot(times, mean_era5, label='ERA5 Truth', marker='o')
plt.plot(times, mean_pred_era5, label='FuXi-ERA5 Init', marker='o')
plt.plot(times, mean_pred_gfs, label='FuXi-GFS Init', marker='o')
plt.xlabel('Forecast Time')
plt.ylabel('z500 Mean')
plt.title('z500 Mean vs Forecast Time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("z500_mean_curve.png")
plt.close()

target_lat = 40.0
target_lon = 116.0

ds_era5 = xr.open_zarr(ERA5_ROOT)
lat_arr = ds_era5['lat'].values
lon_arr = ds_era5['lon'].values
ds_era5.close()

# 找到最近的索引
lat_idx = np.argmin(np.abs(lat_arr - target_lat))
lon_idx = np.argmin(np.abs(lon_arr - target_lon))

z500_true_curve = []
z500_pred_era5_curve = []
z500_pred_gfs_curve = []

for i, step in enumerate(STEPS):
    pred_dir_era5 = os.path.join(PRED_ROOT_ERA5, f"{step}.zarr")
    pred_dir_gfs = os.path.join(PRED_ROOT_TRANS_GFS, f"{step}.zarr")
    pred_era5 = get_pred_z500(pred_dir_era5)
    pred_gfs = get_pred_z500(pred_dir_gfs)
    true = get_true_z500(times[i])
    z500_true_curve.append(true[lat_idx, lon_idx])
    z500_pred_era5_curve.append(pred_era5[lat_idx, lon_idx])
    z500_pred_gfs_curve.append(pred_gfs[lat_idx, lon_idx])

plt.figure(figsize=(10,5))
plt.plot(times, z500_true_curve, label='ERA5 Truth', marker='o', color='black')
plt.plot(times, z500_pred_era5_curve, label='FuXi-ERA5 Init', marker='o', color='blue', linestyle='--')
plt.plot(times, z500_pred_gfs_curve, label='FuXi-GFS Init', marker='o', color='red', linestyle='-.')
plt.xlabel('Forecast Time')
plt.ylabel('z500 at (lat=%.1f, lon=%.1f)' % (lat_arr[lat_idx], lon_arr[lon_idx]))
plt.title('z500 Prediction Curve at (lat=%.1f, lon=%.1f)' % (lat_arr[lat_idx], lon_arr[lon_idx]))
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("z500_point_curve.png")
plt.close()

print("已保存单点z500随时间预测曲线：z500_point_curve.png")

print("评估完成，曲线已保存为 z500_rmse_curve.png、z500_acc_curve.png 和 z500_mean_curve.png")