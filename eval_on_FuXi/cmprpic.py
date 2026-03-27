import os
import argparse

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd


def load_rtm_pred(pred_root: str, step: str = "001") -> xr.DataArray:
    """从 RTM 预报目录读取指定 step 的场，返回 DataArray.

    约定：pred_root/step.zarr 是一个 Zarr store，里面至少有一个变量：
    - 如果只有一个变量，则直接取这个变量；
    - 如有多个变量，可以用 --var_name 指定（在上层处理）。
    """
    store_path = os.path.join(pred_root, f"{step}.zarr")
    if not os.path.exists(store_path):
        raise FileNotFoundError(f"预测文件不存在: {store_path}")

    ds = xr.open_zarr(store_path)
    # 如果只有一个变量，直接取之
    data_vars = list(ds.data_vars)
    if len(data_vars) == 0:
        raise ValueError(f"Zarr 中没有 data_vars: {store_path}")
    var = data_vars[0]
    da = ds[var]

    # 通常 RTM 输出可能是 (time, member, step, channel/level, lat, lon)
    # 这里只取第一个 time / member / step，剩下由上层做 channel/level 选择
    for dim in ["time", "member", "step"]:
        if dim in da.dims:
            da = da.isel({dim: 0})

    return da.load()


def load_gt(era5_root: str, target_time: pd.Timestamp) -> xr.DataArray:
    """从 ERA5 大 zarr（与 pairset.py 相同结构）读取指定时间步的 GT 场.

    约定：era5_root 是一个包含变量 `data` 的 zarr，维度至少包含 (time, channel, lat, lon)。
    这里按时间选出一帧，返回 DataArray: (channel, lat, lon)。
    """
    if not os.path.exists(era5_root):
        raise FileNotFoundError(f"ERA5 根目录不存在: {era5_root}")

    ds = xr.open_zarr(era5_root, consolidated=False)
    if "data" not in ds:
        raise ValueError(f"在 ERA5 zarr 中未找到变量 'data': {era5_root}")

    # 按时间精确匹配目标时间
    try:
        da = ds["data"].sel(time=target_time)
    except Exception:
        times = pd.to_datetime(ds.time.values)
        if target_time not in times:
            raise ValueError(f"ERA5 中找不到时间 {target_time}，可用时间范围为 {times.min()} ~ {times.max()}")
        idx = int(np.where(times == target_time)[0][0])
        da = ds["data"].isel(time=idx)

    # 此时 da 形状通常为 (channel, lat, lon)
    return da.load()


def parse_init_time_from_pred_root(pred_root: str) -> pd.Timestamp:
    """从 pred_root 目录名解析起报时间，例如 '20250101-12' -> 2025-01-01 12:00:00."""
    base = os.path.basename(os.path.normpath(pred_root))
    # 兼容 'YYYYMMDD-12' 或 'YYYYMMDD-1200' 等格式
    if "-" not in base:
        raise ValueError(f"无法从目录名解析起报时间: {base}")
    date_str, hour_str = base.split("-", 1)
    if len(hour_str) == 2:
        dt_str = f"{date_str} {hour_str}:00:00"
    elif len(hour_str) == 4:
        dt_str = f"{date_str} {hour_str[:2]}:{hour_str[2:]}:00"
    else:
        raise ValueError(f"无法解析小时部分: {hour_str}")
    return pd.to_datetime(dt_str)


def select_level_or_channel(da: xr.DataArray, name: str | None) -> xr.DataArray:
    """根据 level 或 channel 选择一个高空层/变量。

    - 如果存在 "level" 维，name 例如 "500" / "500.0"，则按数值选 level。
    - 如果存在 "channel" 维，name 例如 "z500" / "t500"，则按字符串选 channel。
    - 如果 name 为 None 且存在 level/channel，则默认取第一个。
    """
    if "level" in da.dims:
        if name is None:
            return da.isel(level=0)
        try:
            lvl = float(name)
        except ValueError:
            raise ValueError("当数据有 level 维时，--var_name 应该是数值字符串，例如 '500' 表示 500hPa")
        levels = da["level"].values
        idx = int(np.argmin(np.abs(levels - lvl)))
        return da.isel(level=idx)

    if "channel" in da.dims:
        if name is None:
            return da.isel(channel=0)
        chans = np.array([str(c) for c in da["channel"].values])
        if name not in chans:
            raise ValueError(f"channel {name!r} 不在数据中，现有 channel 有: {chans}")
        return da.sel(channel=name)

    # 没有 level/channel，直接返回
    return da


def plot_gt_vs_pred(gt: xr.DataArray, pred: xr.DataArray, title: str, save_path: str | None = None):
    """画 GT vs 预报 的二维对比图（地图形式）+ 差值。"""
    if {"lat", "lon"} <= set(gt.dims):
        lat = gt["lat"].values
        lon = gt["lon"].values
    else:
        # 尝试通用 dim 名
        lat = gt[gt.dims[-2]].values
        lon = gt[gt.dims[-1]].values

    gt_np = gt.values
    pred_np = pred.values
    diff_np = pred_np - gt_np

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), subplot_kw={"projection": None})

    im0 = axes[0].pcolormesh(lon, lat, gt_np, shading="auto")
    axes[0].set_title("GT")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(lon, lat, pred_np, shading="auto")
    axes[1].set_title("Forecast")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(lon, lat, diff_np, shading="auto")
    axes[2].set_title("Forecast - GT")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

    fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"已保存图片到: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="GT vs 预报 对比图绘制")
    parser.add_argument("--pred_root", type=str,
                        default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/huangqiusheng/eval/RTM_base_6h/20250101-12",
                        help="RTM 预报目录 (包含 step.zarr 子目录)")
    parser.add_argument("--gt_root", type=str,
                        default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/huangqiusheng/datasets/era5.rtm.02_25.6h.c109.new3/",
                        help="GT ERA5 大 zarr 根目录 (与 pairset.py 中 ERA5_PATH 相同结构)")
    parser.add_argument("--step", type=str, default="001", help="要对比的预报步 (例如 '001')")
    parser.add_argument("--var_name", type=str, default=None,
                        help="高空变量：对有 level 维时为高度(如 '500')，有 channel 维时为名字(如 'z500')；为空则取第一个")
    parser.add_argument("--save_path", type=str, default="./gt_vs_pred_20250101-12_step001.png",
                        help="输出图片路径")

    args = parser.parse_args()

    print(f"读取预报: {args.pred_root}, step={args.step}")
    pred_da = load_rtm_pred(args.pred_root, args.step)

    # 由 pred_root 目录名解析起报时间，再根据 step 推出目标验证时间
    init_time = parse_init_time_from_pred_root(args.pred_root)
    lead_hours = int(args.step) * 6  # RTM_base_6h: 每个 step 为 6 小时
    target_time = init_time + pd.Timedelta(hours=lead_hours)

    print(f"读取 GT: {args.gt_root}, target_time={target_time}")
    gt_da = load_gt(args.gt_root, target_time)

    # 选定单层/单 channel
    pred_sel = select_level_or_channel(pred_da, args.var_name)
    gt_sel = select_level_or_channel(gt_da, args.var_name)

    # 对齐 lat/lon （如果坐标名相同、值一致，则自动对齐）
    gt_sel, pred_sel = xr.align(gt_sel, pred_sel, join="exact")

    title = f"GT vs Forecast (step {args.step}, var={args.var_name or 'first'})"
    plot_gt_vs_pred(gt_sel, pred_sel, title, args.save_path)


if __name__ == "__main__":
    main()
