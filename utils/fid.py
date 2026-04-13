import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


TARGET_CHANNELS = [
    # 温度（高空13层 + 地表t2m）
    "t50", "t100", "t150", "t200", "t250", "t300", "t400", "t500",
    "t600", "t700", "t850", "t925", "t1000", "t2m",
    # U风（地表u10m + 高空13层）
    "u10m", "u50", "u100", "u150", "u200", "u250", "u300", "u400",
    "u500", "u600", "u700", "u850", "u925", "u1000",
    # V风（地表v10m + 高空13层）
    "v10m", "v50", "v100", "v150", "v200", "v250", "v300", "v400",
    "v500", "v600", "v700", "v850", "v925", "v1000",
    # 位势高度（高空13层）
    "z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500",
    "z600", "z700", "z850", "z925", "z1000",
    # 比湿（高空13层）
    "q50", "q100", "q150", "q200", "q250", "q300", "q400", "q500",
    "q600", "q700", "q850", "q925", "q1000",
    # 地表变量
    "msl", "tp",
]

DEFAULT_DATES = [
    "20250101", "20250115", "20250131", "20250214", "20250301",
    "20250315", "20250331", "20250501", "20250515", "20250601",
]


def _to_str_channels(ch_values: Iterable) -> list[str]:
    out = []
    for c in ch_values:
        if isinstance(c, bytes):
            out.append(c.decode("utf-8").strip())
        else:
            out.append(str(c).strip())
    return out


def _pick_data_var(ds: xr.Dataset) -> str:
    if "data" in ds.data_vars:
        return "data"
    if "output" in ds.data_vars:
        return "output"
    return list(ds.data_vars.keys())[0]


def _compute_stats(
    ds: xr.Dataset,
    data_var: str,
    channels: list[str],
    times: np.ndarray,
    time_chunk: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]:
    da = ds[data_var].sel(time=times, channel=channels)

    c = len(channels)
    sum_vec = np.zeros((c,), dtype=np.float64)
    sum_xx = np.zeros((c, c), dtype=np.float64)
    total_n = 0

    n_time = da.sizes["time"]
    for i in range(0, n_time, time_chunk):
        j = min(i + time_chunk, n_time)
        block = da.isel(time=slice(i, j)).values.astype(np.float64)  # [t, c, h, w]
        x = np.transpose(block, (0, 2, 3, 1)).reshape(-1, c)  # [n, c]

        finite_mask = np.isfinite(x).all(axis=1)
        x = x[finite_mask]
        if x.shape[0] == 0:
            continue

        total_n += x.shape[0]
        sum_vec += x.sum(axis=0)
        sum_xx += x.T @ x

    if total_n == 0:
        raise RuntimeError("没有可用样本（可能全是 NaN/Inf）")

    mean = sum_vec / total_n
    if total_n > 1:
        cov = (sum_xx - np.outer(sum_vec, sum_vec) / total_n) / (total_n - 1)
    else:
        cov = np.zeros((c, c), dtype=np.float64)

    return mean, cov, total_n


def _trace_sqrt_product(cov1: np.ndarray, cov2: np.ndarray, eps: float = 1e-6) -> float:
    c = cov1.shape[0]
    cov1 = cov1 + eps * np.eye(c, dtype=np.float64)
    cov2 = cov2 + eps * np.eye(c, dtype=np.float64)

    eigvals1, eigvecs1 = np.linalg.eigh((cov1 + cov1.T) * 0.5)
    eigvals1 = np.clip(eigvals1, 0.0, None)
    sqrt_cov1 = eigvecs1 @ np.diag(np.sqrt(eigvals1)) @ eigvecs1.T

    m = sqrt_cov1 @ cov2 @ sqrt_cov1
    m = (m + m.T) * 0.5
    eigvals_m = np.linalg.eigvalsh(m)
    eigvals_m = np.clip(eigvals_m, 0.0, None)
    return float(np.sum(np.sqrt(eigvals_m)))


def fid_from_stats(mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray) -> float:
    diff = mu1 - mu2
    trace_sqrt = _trace_sqrt_product(cov1, cov2)
    fid = float(diff @ diff + np.trace(cov1) + np.trace(cov2) - 2.0 * trace_sqrt)
    return max(fid, 0.0)


def compute_fid_for_pair(
    pred_zarr: str,
    era5_root: str,
    channels_mode: str,
    time_chunk: int,
) -> dict:
    pred_ds = xr.open_zarr(pred_zarr, consolidated=False)
    era5_ds = xr.open_zarr(era5_root, consolidated=False)

    pred_var = _pick_data_var(pred_ds)
    era5_var = _pick_data_var(era5_ds)

    pred_times = pd.to_datetime(pred_ds["time"].values)
    era5_times = pd.to_datetime(era5_ds["time"].values)
    common_times = np.intersect1d(pred_times.values.astype("datetime64[ns]"), era5_times.values.astype("datetime64[ns]"))
    if common_times.size == 0:
        raise RuntimeError(f"没有共同时间步: {pred_zarr}")

    pred_channels = _to_str_channels(pred_ds["channel"].values)
    era5_channels = _to_str_channels(era5_ds["channel"].values)
    common_channels = [c for c in pred_channels if c in set(era5_channels)]

    if channels_mode == "c109":
        channels = common_channels
    elif channels_mode == "target70":
        channels = [c for c in TARGET_CHANNELS if c in set(common_channels)]
    else:
        raise ValueError(f"unknown channels_mode: {channels_mode}")

    if len(channels) == 0:
        raise RuntimeError("没有可用通道")

    mu_p, cov_p, n_p = _compute_stats(pred_ds, pred_var, channels, common_times, time_chunk=time_chunk)
    mu_e, cov_e, n_e = _compute_stats(era5_ds, era5_var, channels, common_times, time_chunk=time_chunk)
    fid = fid_from_stats(mu_p, cov_p, mu_e, cov_e)

    pred_ds.close()
    era5_ds.close()

    return {
        "pred_zarr": pred_zarr,
        "channels_mode": channels_mode,
        "num_channels": len(channels),
        "num_common_times": int(common_times.size),
        "num_samples_pred": int(n_p),
        "num_samples_era5": int(n_e),
        "fid": float(fid),
    }


def find_pred_zarrs(root_dir: str, dates: list[str]) -> list[str]:
    root = Path(root_dir)
    found = []
    for d in dates:
        cands = sorted(root.glob(f"*{d}*/era5_localreplaced.zarr"))
        if len(cands) == 0:
            print(f"[WARN] 未找到日期 {d} 对应的 zarr")
            continue
        found.append(str(cands[-1]))
    return found


def main():
    parser = argparse.ArgumentParser(description="Compute FID between local-replaced c109 zarr and ERA5 zarr")
    parser.add_argument(
        "--era5_root",
        type=str,
        default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/huangqiusheng/datasets/era5.rtm.02_25.6h.c109.new3/",
    )
    parser.add_argument(
        "--single_pred_zarr",
        type=str,
        default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/localreplaced/with_trans_gfs/swinunet_2022_2024_3yr_L1+Gradloss_4_6_20250515/era5_localreplaced.zarr",
    )
    parser.add_argument(
        "--pred_root",
        type=str,
        default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/localreplaced/with_trans_gfs",
        help="根目录；会按日期自动查找 *<date>*/era5_localreplaced.zarr",
    )
    parser.add_argument("--dates", type=str, nargs="+", default=DEFAULT_DATES)
    parser.add_argument("--time_chunk", type=int, default=1)
    parser.add_argument("--output_csv", type=str, default="/home/ximutian/fid_results.csv")
    args = parser.parse_args()

    tasks = []
    if args.single_pred_zarr:
        tasks.append(args.single_pred_zarr)

    auto_found = find_pred_zarrs(args.pred_root, args.dates)
    for p in auto_found:
        if p not in tasks:
            tasks.append(p)

    if len(tasks) == 0:
        raise RuntimeError("未找到任何待评估的预测 zarr")

    results = []
    for pred_zarr in tasks:
        print(f"\n===== 评估: {pred_zarr} =====")
        for mode in ["c109", "target70"]:
            r = compute_fid_for_pair(
                pred_zarr=pred_zarr,
                era5_root=args.era5_root,
                channels_mode=mode,
                time_chunk=args.time_chunk,
            )
            print(
                f"[{mode}] FID={r['fid']:.6f}, channels={r['num_channels']}, "
                f"common_times={r['num_common_times']}"
            )
            results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\n结果已保存: {args.output_csv}")


if __name__ == "__main__":
    main()
