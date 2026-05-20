import os
import sys
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm
from fuxi.fuxi_grad import UTransformer, FuXi, time_encoding

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

# ============================================================
# Constants
# ============================================================
DEFAULT_DATES = [
    "20250101", "20250115", "20250201", "20250215", "20250301", "20250315",
    "20250401", "20250415", "20250501", "20250515", "20250601", "20250615",
    "20250701", "20250715", "20250801", "20250815", "20250901", "20250915",
    "20251001", "20251015", "20251101",
]

CHANNELS = [
    'z50', 'z100', 'z150', 'z200', 'z250', 'z300', 'z400', 'z500',
    'z600', 'z700', 'z850', 'z925', 'z1000', 't50', 't100', 't150',
    't200', 't250', 't300', 't400', 't500', 't600', 't700', 't850',
    't925', 't1000', 'u50', 'u100', 'u150', 'u200', 'u250', 'u300',
    'u400', 'u500', 'u600', 'u700', 'u850', 'u925', 'u1000', 'v50',
    'v100', 'v150', 'v200', 'v250', 'v300', 'v400', 'v500', 'v600',
    'v700', 'v850', 'v925', 'v1000', 'r50', 'r100', 'r150', 'r200',
    'r250', 'r300', 'r400', 'r500', 'r600', 'r700', 'r850', 'r925',
    'r1000', 't2m', 'u10m', 'v10m', 'msl', 'tp',
]

Z500_IDX = CHANNELS.index("z500")  # 7
TP_IDX = CHANNELS.index("tp")
FORECAST_STEPS = 40
HOURS_PER_STEP = 6

# Default paths (server paths, override via CLI)
DEFAULT_GFS_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/gfs_2020_2025_c226_normalized"
DEFAULT_ERA5_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanjiang/dataset/era5.2010_2025.c226.zarr"
DEFAULT_FUXI_DIR = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/fuxi_inference/main/fuxi"
DEFAULT_G2E_CKPT = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/checkpoints/swinunet_align_0423/checkpoint_epoch_146.pth"
DEFAULT_CLIM_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanjiang/eval/era5/clim.daily"
DEFAULT_OUTPUT_DIR = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/inference_align_results"


# ============================================================
# Model builders (mirrors main_align.py)
# ============================================================
def build_g2e_model(device: torch.device, checkpoint_path: str):
    from models.swinUNET import G2E

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
        using_checkpoints=False,
        using_time_embedding=True,
        res_per_stage=[1, 1, 1],
        channels=[384, 768, 1536],
        using_kl=False,
        dropout_rate=0.0,
        use_skip_connections=True,
        use_residual_blocks=True,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("_orig_mod.", "").replace("module.", "")
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


def build_fuxi_model(device: torch.device, fuxi_dir: str):
    sys.path.insert(0, fuxi_dir)
    

    conds = np.load(os.path.join(fuxi_dir, "conds.npy"))
    std = np.load(os.path.join(fuxi_dir, "std.npy"))
    mean = np.load(os.path.join(fuxi_dir, "mean.npy"))

    const = torch.from_numpy(conds).to(device=device, dtype=torch.float32)
    std_t = torch.from_numpy(std).to(device=device, dtype=torch.float32)
    mean_t = torch.from_numpy(mean).to(device=device, dtype=torch.float32)

    decoder = UTransformer(
        in_chans=75,
        out_chans=70,
        in_frames=2,
        image_size=(720, 1440),
        window_size=9,
        patch_size=4,
        down_times=1,
        embed_dim=1536,
        num_heads=24,
        depths=[12, 12, 12, 12],
    )

    model = FuXi(
        in_frames=2,
        out_frames=1,
        step_range=[FORECAST_STEPS],
        decoder=[decoder, decoder, decoder],
        const=const,
        std=std_t,
        mean=mean_t,
        device=str(device),
        dtype=torch.float32,
    ).to(device=device, dtype=torch.float32)

    model.load(fuxi_dir, fmt="pth")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


# ============================================================
# Data I/O helpers
# ============================================================
def _decode_channel_values(values) -> list:
    out = []
    for v in values:
        if isinstance(v, bytes):
            out.append(v.decode())
        else:
            out.append(str(v))
    return out


def _get_channel_dim(ds: xr.Dataset) -> str:
    if "channel" in ds.dims or "channel" in ds.coords:
        return "channel"
    if "level" in ds.dims or "level" in ds.coords:
        return "level"
    raise KeyError(f"Dataset missing channel/level dim. dims: {list(ds.dims)}")


def _open_dataarray_robust(path: str) -> xr.DataArray:
    """Robustly open mean.nc / std.nc (mirrors fuxi_rmse_interface.py)."""
    try:
        return xr.open_dataarray(path)
    except Exception:
        ds = xr.open_dataset(path)
        if len(ds.data_vars) == 0:
            raise ValueError(f"No data_vars found in {path}")
        return ds[list(ds.data_vars)[0]]


def _to_channel_first_stats(arr: np.ndarray, expected_c: int) -> np.ndarray:
    """Ensure stats are [C, 1, 1] or [C, H, W] (mirrors fuxi_rmse_interface.py)."""
    if arr.ndim == 1:
        return arr[:, None, None]
    if arr.ndim == 3:
        if arr.shape[0] == expected_c:
            return arr
        if arr.shape[-1] == expected_c:
            return np.transpose(arr, (2, 0, 1))
    raise ValueError(f"Unsupported stats shape {arr.shape}, expected channel dim={expected_c}")


def load_era5_stats(era5_zarr_path: str, channels: list):
    """Load ERA5 mean/std, return [C, H, W] or [C, 1, 1] tensors (no batch dim)."""
    mean_da = _open_dataarray_robust(os.path.join(era5_zarr_path, "mean.nc"))
    std_da = _open_dataarray_robust(os.path.join(era5_zarr_path, "std.nc"))

    if "channel" in mean_da.dims:
        mean_da = mean_da.sel(channel=channels)
        std_da = std_da.sel(channel=channels)

    mean_np = mean_da.values.astype(np.float32)
    std_np = std_da.values.astype(np.float32)

    mean_np = _to_channel_first_stats(mean_np, expected_c=len(channels))
    std_np = _to_channel_first_stats(std_np, expected_c=len(channels))

    return torch.from_numpy(mean_np), torch.from_numpy(std_np)


def read_gfs(gfs_zarr_path: str, ts: pd.Timestamp, channels: list) -> torch.Tensor:
    ds = xr.open_zarr(gfs_zarr_path, consolidated=False)
    chan_dim = _get_channel_dim(ds)
    gfs_channels = _decode_channel_values(ds[chan_dim].values)
    chan_indices = [gfs_channels.index(ch) for ch in channels]
    data = ds["data"].sel(time=ts).isel({chan_dim: chan_indices}).values.astype(np.float32)
    ds.close()
    return torch.from_numpy(data)


def read_era5_normalized(era5_zarr_path: str, ts: pd.Timestamp, channels: list) -> torch.Tensor:
    ds = xr.open_zarr(era5_zarr_path, consolidated=False)
    chan_dim = _get_channel_dim(ds)
    era5_channels = _decode_channel_values(ds[chan_dim].values)
    chan_indices = [era5_channels.index(ch) for ch in channels]
    data = ds["data"].sel(time=ts).isel({chan_dim: chan_indices}).values.astype(np.float32)
    ds.close()
    return torch.from_numpy(data)


def denormalize(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Denormalize [C, H, W] or [B, C, H, W] using stats [C, 1, 1] or [C, H, W].

    Mirrors rmse.py per-channel logic:  arr = arr * std[c] + mean[c]
    """
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if data.ndim == 4 and mean.ndim == 3:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    out = data * std + mean

    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# Metrics (mirrors metrics.py)
# ============================================================
def load_climatology(clim_path: str):
    return xr.open_zarr(clim_path)


def get_clim_z500(clim_ds, ts: pd.Timestamp) -> torch.Tensor:
    doy = ts.dayofyear
    hour = ts.hour
    z500_clim = clim_ds["z500"].sel(doy=doy, hour=hour).values
    return torch.from_numpy(z500_clim.astype(np.float32))


def compute_rmse(pred: torch.Tensor, truth: torch.Tensor, lat_weights: torch.Tensor) -> torch.Tensor:
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    truth = torch.nan_to_num(truth, nan=0.0, posinf=0.0, neginf=0.0)
    err2 = (pred - truth) ** 2
    w = lat_weights.expand_as(err2)
    weighted_mse = (err2 * w).sum() / (w.sum() + 1e-12)
    return torch.sqrt(weighted_mse + 1e-12)


def compute_acc(
    pred: torch.Tensor,
    truth: torch.Tensor,
    clim_mean: torch.Tensor,
    lat_weights_norm: torch.Tensor,
) -> torch.Tensor:
    pred_anom = pred - clim_mean
    truth_anom = truth - clim_mean
    w = lat_weights_norm  # [H, 1] broadcasts correctly with [H, W]

    A = (w * pred_anom * truth_anom).sum()
    B = (w * pred_anom ** 2).sum()
    C = (w * truth_anom ** 2).sum()

    return A / torch.sqrt(B * C + 1e-12)


# ============================================================
# Main inference logic
# ============================================================
def run_inference_for_date(
    init_time: pd.Timestamp,
    g2e_model: torch.nn.Module,
    fuxi_model,
    era5_mean: torch.Tensor,
    era5_std: torch.Tensor,
    lat_weights: torch.Tensor,
    lat_weights_norm: torch.Tensor,
    clim_ds,
    gfs_dir: str,
    era5_dir: str,
    device: torch.device,
) -> tuple:
    """Run G2E+FuXi 40-step inference for a single init time.

    Returns (rmse_list, acc_list) each of length FORECAST_STEPS.
    """
    

    # 1. Read GFS at init time (normalized)
    gfs_norm = read_gfs(gfs_dir, init_time, CHANNELS).to(device)

    # 2. Read ERA5 at t-6h (for FuXi cold-start input)
    t_prev = init_time - pd.Timedelta(hours=6)
    era5_prev_norm = read_era5_normalized(era5_dir, t_prev, CHANNELS).to(device)

    # 3. G2E forward: GFS -> ERA5-like (normalized)
    with torch.no_grad():
        g2e_output_norm = g2e_model(
            gfs_norm.unsqueeze(0),
            times=np.array([str(init_time)]),
        )
    g2e_output_norm = g2e_output_norm.squeeze(0)

    # 4. Denormalize both to physical units for FuXi
    era5_prev_phys = denormalize(era5_prev_norm, era5_mean, era5_std)
    g2e_output_phys = denormalize(g2e_output_norm, era5_mean, era5_std)

    # ---- debug: check Z500 physical values ----
    _z500_mean_val = era5_mean[Z500_IDX].item() if era5_mean.ndim == 1 else float(era5_mean[Z500_IDX].mean())
    _z500_std_val = era5_std[Z500_IDX].item() if era5_std.ndim == 1 else float(era5_std[Z500_IDX].mean())
    print(f"  [DEBUG] era5_mean[Z500]~{_z500_mean_val:.1f}, era5_std[Z500]~{_z500_std_val:.1f}")
    print(f"  [DEBUG] era5_prev z500: min={era5_prev_phys[Z500_IDX].min().item():.1f}, "
          f"max={era5_prev_phys[Z500_IDX].max().item():.1f}, "
          f"mean={era5_prev_phys[Z500_IDX].mean().item():.1f}")
    print(f"  [DEBUG] g2e_out   z500: min={g2e_output_phys[Z500_IDX].min().item():.1f}, "
          f"max={g2e_output_phys[Z500_IDX].max().item():.1f}, "
          f"mean={g2e_output_phys[Z500_IDX].mean().item():.1f}")

    # 5. Stack FuXi input: [ERA5(t-6h), G2E(t0)]
    fuxi_input = torch.stack([era5_prev_phys, g2e_output_phys], dim=0)

    # 6. Time encoding for 40 steps
    tembs = time_encoding(init_time, FORECAST_STEPS, freq=HOURS_PER_STEP)
    tembs = tembs.to(device=device, dtype=torch.float32)

    # 7. FuXi 40-step forecast
    with torch.no_grad():
        # FuXi forward returns [1, 40, 70, 721, 1440]
        outputs = fuxi_model.forward((fuxi_input, tembs))
    outputs = outputs.squeeze(0)  # [40, 70, 721, 1440]

    # 8. Per-step Z500 metrics
    rmse_list, acc_list = [], []

    pbar_steps = tqdm(range(FORECAST_STEPS), desc=f"  Steps for {init_time.strftime('%Y%m%d')}", leave=False)
    for step in pbar_steps:
        lead_hours = (step + 1) * HOURS_PER_STEP
        target_time = init_time + pd.Timedelta(hours=lead_hours)

        try:
            era5_truth_norm = read_era5_normalized(era5_dir, target_time, CHANNELS).to(device)
        except Exception:
            rmse_list.append(np.nan)
            acc_list.append(np.nan)
            pbar_steps.set_postfix({"status": "truth_missing"})
            continue

        era5_truth_phys = denormalize(era5_truth_norm, era5_mean, era5_std)

        z500_pred = outputs[step, Z500_IDX]
        z500_truth = era5_truth_phys[Z500_IDX]

        # debug: first step Z500 comparison
        if step == 0:
            print(f"  [DEBUG] Step1 FuXi z500: min={z500_pred.min().item():.1f}, "
                  f"max={z500_pred.max().item():.1f}, mean={z500_pred.mean().item():.1f}")
            print(f"  [DEBUG] Step1 Truth z500: min={z500_truth.min().item():.1f}, "
                  f"max={z500_truth.max().item():.1f}, mean={z500_truth.mean().item():.1f}")

        rmse = compute_rmse(z500_pred, z500_truth, lat_weights)
        rmse_val = float(rmse.cpu())
        rmse_list.append(rmse_val)

        clim_mean = get_clim_z500(clim_ds, target_time).to(device)
        acc = compute_acc(z500_pred, z500_truth, clim_mean, lat_weights_norm)
        acc_val = float(acc.cpu())
        acc_list.append(acc_val)
        
        # Update step progress bar with metrics
        pbar_steps.set_postfix({"step": step + 1, "rmse": f"{rmse_val:.4f}", "acc": f"{acc_val:.4f}"})

    return rmse_list, acc_list


def write_date_results(output_dir: str, date_str: str, rmse_list: list, acc_list: list):
    txt_path = os.path.join(output_dir, f"{date_str}_z500.txt")
    with open(txt_path, "w") as f:
        f.write(f"# Init: {date_str} 00Z  |  Steps: {FORECAST_STEPS}x{HOURS_PER_STEP}h\n")
        f.write(f"# {'Step':>6s}  {'Lead(h)':>8s}  {'RMSE':>10s}  {'ACC':>10s}\n")
        for step in range(FORECAST_STEPS):
            lead_h = (step + 1) * HOURS_PER_STEP
            rmse_str = f"{rmse_list[step]:.4f}" if not np.isnan(rmse_list[step]) else "N/A"
            acc_str = f"{acc_list[step]:.4f}" if not np.isnan(acc_list[step]) else "N/A"
            f.write(f"  {step+1:>4d}  {lead_h:>8d}  {rmse_str:>10s}  {acc_str:>10s}\n")


def write_summary(output_dir: str, all_results: dict):
    """Write a summary file with key steps across all dates."""
    summary_path = os.path.join(output_dir, "summary_all_dates.txt")
    key_steps = [0, 9, 19, 29, 39]  # step 1, 10, 20, 30, 40
    with open(summary_path, "w") as f:
        header = (
            f"# {'Date':>10s}"
            + "".join(f"  {'S{}_{}h_RMSE'.format(s+1, (s+1)*6):>14s}  {'S{}_{}h_ACC'.format(s+1, (s+1)*6):>12s}" for s in key_steps)
            + "\n"
        )
        f.write(header)
        for date_str in sorted(all_results.keys()):
            rmse_list, acc_list = all_results[date_str]
            line = f"  {date_str:>10s}"
            for s in key_steps:
                if s < len(rmse_list):
                    line += f"  {rmse_list[s]:14.4f}  {acc_list[s]:12.4f}"
                else:
                    line += f"  {'N/A':>14s}  {'N/A':>12s}"
            line += "\n"
            f.write(line)


# ============================================================
# Entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="G2E + FuXi Align Inference (40-step forecast)")
    parser.add_argument("--gfs_dir", type=str, default=DEFAULT_GFS_PATH)
    parser.add_argument("--era5_dir", type=str, default=DEFAULT_ERA5_PATH)
    parser.add_argument("--fuxi_dir", type=str, default=DEFAULT_FUXI_DIR)
    parser.add_argument("--g2e_ckpt", type=str, default=DEFAULT_G2E_CKPT)
    parser.add_argument("--clim_path", type=str, default=DEFAULT_CLIM_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dates", type=str, nargs="+", default=DEFAULT_DATES)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_g2e", action="store_true",
                        help="Skip G2E, use raw ERA5 as FuXi init (for pure FuXi eval)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Build models ----
    if not args.skip_g2e:
        print("Building G2E model...")
        g2e_model = build_g2e_model(device, args.g2e_ckpt)
        print(f"G2E loaded from {args.g2e_ckpt}")
    else:
        g2e_model = None
        print("Skipping G2E, will use raw ERA5 as FuXi input.")

    print("Building FuXi model...")
    fuxi_model = build_fuxi_model(device, args.fuxi_dir)
    print("FuXi loaded.")

    # ---- Load ERA5 stats ----
    era5_mean, era5_std = load_era5_stats(args.era5_dir, CHANNELS)
    era5_mean = era5_mean.to(device)
    era5_std = era5_std.to(device)

    # ---- Latitude weights ----
    lat = np.linspace(90, -90, 721)
    lat_w = np.cos(np.deg2rad(np.abs(lat))).astype(np.float32)
    lat_weights = torch.from_numpy(lat_w).to(device).view(-1, 1)
    lat_weights_norm = lat_weights / lat_weights.mean()

    # ---- Climatology ----
    print("Loading climatology...")
    clim_ds = load_climatology(args.clim_path)

    # ---- Run inference ----
    all_results = {}
    print(f"\nStarting inference for {len(args.dates)} dates...\n")

    for date_str in tqdm(args.dates, desc="Dates"):
        init_time = pd.Timestamp(f"{date_str} 00:00:00")

        try:
            if args.skip_g2e:
                # Pure FuXi eval: use ERA5(t-6h) and ERA5(t0) as input

                t_prev = init_time - pd.Timedelta(hours=6)
                era5_prev_norm = read_era5_normalized(args.era5_dir, t_prev, CHANNELS).to(device)
                era5_curr_norm = read_era5_normalized(args.era5_dir, init_time, CHANNELS).to(device)

                era5_prev_phys = denormalize(era5_prev_norm, era5_mean, era5_std)
                era5_curr_phys = denormalize(era5_curr_norm, era5_mean, era5_std)

                fuxi_input = torch.stack([era5_prev_phys, era5_curr_phys], dim=0)
                tembs = time_encoding(init_time, FORECAST_STEPS, freq=HOURS_PER_STEP)
                tembs = tembs.to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    outputs = fuxi_model.forward((fuxi_input, tembs))
                outputs = outputs.squeeze(0)

                rmse_list, acc_list = [], []
                pbar_steps = tqdm(range(FORECAST_STEPS), desc=f"  Steps for {init_time.strftime('%Y%m%d')}", leave=False)
                for step in pbar_steps:
                    lead_hours = (step + 1) * HOURS_PER_STEP
                    target_time = init_time + pd.Timedelta(hours=lead_hours)
                    try:
                        era5_truth_norm = read_era5_normalized(args.era5_dir, target_time, CHANNELS).to(device)
                    except Exception:
                        rmse_list.append(np.nan)
                        acc_list.append(np.nan)
                        pbar_steps.set_postfix({"status": "truth_missing"})
                        continue
                    era5_truth_phys = denormalize(era5_truth_norm, era5_mean, era5_std)
                    z500_pred = outputs[step, Z500_IDX]
                    z500_truth = era5_truth_phys[Z500_IDX]
                    rmse = compute_rmse(z500_pred, z500_truth, lat_weights)
                    rmse_val = float(rmse.cpu())
                    rmse_list.append(rmse_val)
                    clim_mean = get_clim_z500(clim_ds, target_time).to(device)
                    acc = compute_acc(z500_pred, z500_truth, clim_mean, lat_weights_norm)
                    acc_val = float(acc.cpu())
                    acc_list.append(acc_val)
                    pbar_steps.set_postfix({"step": step + 1, "rmse": f"{rmse_val:.4f}", "acc": f"{acc_val:.4f}"})
            else:
                rmse_list, acc_list = run_inference_for_date(
                    init_time, g2e_model, fuxi_model,
                    era5_mean, era5_std, lat_weights, lat_weights_norm, clim_ds,
                    args.gfs_dir, args.era5_dir, device,
                )

            all_results[date_str] = (rmse_list, acc_list)
            write_date_results(args.output_dir, date_str, rmse_list, acc_list)

            s1_rmse = rmse_list[0] if not np.isnan(rmse_list[0]) else float("nan")
            s1_acc = acc_list[0] if not np.isnan(acc_list[0]) else float("nan")
            s40_rmse = rmse_list[-1] if not np.isnan(rmse_list[-1]) else float("nan")
            s40_acc = acc_list[-1] if not np.isnan(acc_list[-1]) else float("nan")
            print(f"  {date_str}: Step1 RMSE={s1_rmse:.4f} ACC={s1_acc:.4f}  "
                  f"Step40 RMSE={s40_rmse:.4f} ACC={s40_acc:.4f}")

        except Exception as e:
            print(f"  {date_str}: FAILED - {e}")
            import traceback
            traceback.print_exc()

    # ---- Summary ----
    if all_results:
        write_summary(args.output_dir, all_results)
        print(f"\nAll results saved to {args.output_dir}")
        print(f"Summary: {os.path.join(args.output_dir, 'summary_all_dates.txt')}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
