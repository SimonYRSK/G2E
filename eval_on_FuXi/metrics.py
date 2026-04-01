import os
import numpy as np
import xarray as xr
import pandas as pd
import zarr
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import argparse

PRED_ROOT_RTM = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/huangqiusheng/eval/RTM_base_6h/20240101-12"
PRED_ROOT_ERA5 = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/era5/20250315-12"
PRED_ROOT_NAIVE_GFS = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/inference_naive_gfs/20250315-12"
PRED_ROOT_TRANS_GFS = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/inference_trans_gfs/swinunet_2022_2024_3yr_3_25_20250315/20250315-12"
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

def main():
    global ERA5_ROOT, TARGET_CHANNEL
    parser = argparse.ArgumentParser(description="Evaluate ACC/RMSE for z500")
    parser.add_argument("--pred_root_era5", type=str, default=PRED_ROOT_ERA5)
    parser.add_argument("--pred_root_naive_gfs", type=str, default=PRED_ROOT_NAIVE_GFS)
    parser.add_argument("--pred_root_trans_gfs", type=str, default=PRED_ROOT_TRANS_GFS)
    parser.add_argument("--era5_root", type=str, default=ERA5_ROOT)
    parser.add_argument("--target_channel", type=str, default=TARGET_CHANNEL)
    parser.add_argument("--start_time", type=str, default="2025-03-15 12:00:00")
    parser.add_argument("--hour_interval", type=int, default=6)
    parser.add_argument("--clim_path", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanjiang/eval/era5/clim.daily")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--tag", type=str, default="3yr")
    parser.add_argument("--date_tag", type=str, default="20250315")
    parser.add_argument("--sanity_check", action="store_true")
    args = parser.parse_args()

    ERA5_ROOT = args.era5_root
    TARGET_CHANNEL = args.target_channel

    print(f"pred_root_era5: {args.pred_root_era5}")
    print(f"pred_root_naive_gfs: {args.pred_root_naive_gfs}")
    print(f"pred_root_trans_gfs: {args.pred_root_trans_gfs}")

    steps = get_steps(args.pred_root_trans_gfs)
    start_time = pd.Timestamp(args.start_time)
    hour_interval = args.hour_interval
    times = [start_time + pd.Timedelta(hours=hour_interval * int(step)) for step in steps]
    clim = xr.open_zarr(args.clim_path)

    if args.sanity_check:
        try:
            if os.path.samefile(args.pred_root_naive_gfs, args.pred_root_trans_gfs):
                print("⚠️ naive_gfs 与 trans_gfs 目录相同，结果会完全一致")
        except Exception:
            pass

        if steps:
            step0 = steps[0]
            naive_dir = os.path.join(args.pred_root_naive_gfs, f"{step0}.zarr")
            trans_dir = os.path.join(args.pred_root_trans_gfs, f"{step0}.zarr")
            try:
                naive = get_pred_z500(naive_dir, pd.Timestamp(args.start_time))
                trans = get_pred_z500(trans_dir, pd.Timestamp(args.start_time))
                diff = (naive - trans).values
                print(
                    f"sanity_check step {step0}: "
                    f"naive[min={float(naive.min()):.3f}, max={float(naive.max()):.3f}, mean={float(naive.mean()):.3f}] "
                    f"trans[min={float(trans.min()):.3f}, max={float(trans.max()):.3f}, mean={float(trans.mean()):.3f}] "
                    f"diff_abs_max={float(np.nanmax(np.abs(diff))):.6f}"
                )
            except Exception as e:
                print(f"sanity_check 失败: {e}")

    rmse_era5 = []
    rmse_naive_gfs = []
    rmse_trans_gfs = []
    acc_era5 = []
    acc_naive_gfs = []
    acc_trans_gfs = []

    log_lines = []

    for i, step in enumerate(steps):
        trans_gfs_dir = os.path.join(args.pred_root_trans_gfs, f"{step}.zarr")
        naive_gfs_dir = os.path.join(args.pred_root_naive_gfs, f"{step}.zarr")
        era5_dir = os.path.join(args.pred_root_era5, f"{step}.zarr")
        if not (os.path.exists(trans_gfs_dir) and os.path.exists(naive_gfs_dir) and os.path.exists(era5_dir)):
            print(f"跳过 step {step}: 预测或真值缺失")
            continue
        pred_trans_gfs = get_pred_z500(trans_gfs_dir, times[i])
        pred_naive_gfs = get_pred_z500(naive_gfs_dir, times[i])
        pred_era5 = get_pred_z500(era5_dir, times[i])
        true = get_true_z500(times[i])

        rmse_trans_gfs.append(calc_rmse(pred_trans_gfs, true))
        rmse_naive_gfs.append(calc_rmse(pred_naive_gfs, true))
        rmse_era5.append(calc_rmse(pred_era5, true))

        acc_trans_gfs.append(compute_acc(pred_trans_gfs, true, clim))
        acc_naive_gfs.append(compute_acc(pred_naive_gfs, true, clim))
        acc_era5.append(compute_acc(pred_era5, true, clim))

        line = (
            f"Step {step}:Naive ERA5 RMSE={rmse_era5[-1]:.3f}, ACC={acc_era5[-1]:.3f} | "
            f"Naive GFS RMSE={rmse_naive_gfs[-1]:.3f}, ACC={acc_naive_gfs[-1]:.3f} | "
            f"GFS2ERA5 RMSE={rmse_trans_gfs[-1]:.3f}, ACC={acc_trans_gfs[-1]:.3f}"
        )
        print(line)
        log_lines.append(line)

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"metrics_{args.tag}_{args.date_tag}.csv")
    out_txt = os.path.join(args.output_dir, f"metrics_{args.tag}_{args.date_tag}.txt")

    df = pd.DataFrame({
        "step": [int(s) for s in steps],
        "rmse_era5": rmse_era5,
        "rmse_naive_gfs": rmse_naive_gfs,
        "rmse_trans_gfs": rmse_trans_gfs,
        "acc_era5": acc_era5,
        "acc_naive_gfs": acc_naive_gfs,
        "acc_trans_gfs": acc_trans_gfs,
    })
    df.to_csv(out_csv, index=False)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    acc_png = os.path.join(args.output_dir, f"z500_acc_rtm_curve_{args.tag}_{args.date_tag}.png")
    rmse_png = os.path.join(args.output_dir, f"z500_rmse_rtm_curve_{args.tag}_{args.date_tag}.png")

    plt.figure(figsize=(10,5))
    plt.plot([int(s) for s in steps], acc_era5, label='Naive_ERA5 ACC', marker='o')
    plt.plot([int(s) for s in steps], acc_naive_gfs, label='Naive_GFS ACC', marker='o')
    plt.plot([int(s) for s in steps], acc_trans_gfs, label='GFS_2_ERA5 ACC', marker='o')
    plt.xlabel('Forecast Step')
    plt.ylabel('ACC')
    plt.title('ACC (z500)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(acc_png)
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot([int(s) for s in steps], rmse_era5, label='Naive_ERA5 RMSE', marker='o')
    plt.plot([int(s) for s in steps], rmse_naive_gfs, label='Naive_GFS RMSE', marker='o')
    plt.plot([int(s) for s in steps], rmse_trans_gfs, label='GFS_2_ERA5 RMSE', marker='o')
    plt.xlabel('Forecast Step')
    plt.ylabel('RMSE')
    plt.title('RMSE (z500)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(rmse_png)
    plt.close()


if __name__ == "__main__":
    main()
