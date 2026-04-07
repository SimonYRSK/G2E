import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd, title):
    print(f"\n=== {title} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Batch pipeline: inference -> replace -> infer_onnx -> metrics")
    parser.add_argument("--model_tag", type=str, default="3yr_L1+Gradloss", choices=["3yr_L2", "1yr_L2", "1yr_L1+Gradloss", "3yr_L1+Gradloss"])
    parser.add_argument(
        "--dates",
        type=str,
        nargs="+",
        default=["20250101", "20250115","20250131", "20250214","20250301", "20250315","20250331", "20250501", "20250515", "20250601"],
    )
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--checkpoint_path", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/checkpoints/swinunetL1+GRAD_2022_2024_3yr_4_3/checkpoint_epoch_32.pth")
    parser.add_argument("--model_suffix", type=str, default="4_3")
    parser.add_argument("--x_path", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/gfs_2025_c70_normalized")
    parser.add_argument("--gfs_path", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/gfs_2025_c70_normalized")

    parser.add_argument("--inference_root", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/inferenced")
    parser.add_argument("--localreplaced_root", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/localreplaced/with_trans_gfs")
    parser.add_argument("--infer_onnx_root", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/inference_trans_gfs")

    parser.add_argument("--era5_root", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/huangqiusheng/datasets/era5.rtm.02_25.6h.c109.new3/")
    parser.add_argument("--pred_root_era5_base", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/era5")
    parser.add_argument("--pred_root_naive_gfs_base", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/inference_naive_gfs")

    parser.add_argument("--metrics_output_root", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/metrics")
    parser.add_argument("--start_hour", type=int, default=12)
    parser.add_argument("--hour_interval", type=int, default=6)
    parser.add_argument("--clim_path", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanjiang/eval/era5/clim.daily")
    parser.add_argument("--target_channel", type=str, default="z500")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    g2e_dir = repo_root / "G2E"
    rtm_dir = repo_root / "fuxi-rtm" / "fuxi-rtm"

    inference_py = g2e_dir / "inference.py"
    replace_py = g2e_dir / "utils" / "replace.py"
    infer_onnx_py = rtm_dir / "infer_onnx_release.py"
    metrics_py = g2e_dir / "eval_on_FuXi" / "metrics.py"

    for date in args.dates:
        model_name = f"swinunet_2022_2024_{args.model_tag}_{args.model_suffix}_{date}"
        start_time = f"{date} 00:00:00"
        end_time = f"{date} 18:00:00"

        inference_save_path = f"{args.inference_root}/{model_name}"
        replace_output_root = f"{args.localreplaced_root}/{model_name}"
        infer_onnx_save_dir = f"{args.infer_onnx_root}/swinunet_2022_2024_{args.model_tag}_{args.model_suffix}"

        pred_root_era5 = f"{args.pred_root_era5_base}/{date}-{args.start_hour:02d}"
        pred_root_naive_gfs = f"{args.pred_root_naive_gfs_base}/{date}-{args.start_hour:02d}"
        pred_root_trans_gfs = f"{infer_onnx_save_dir}/{date}-{args.start_hour:02d}"

        metrics_output_dir = f"{args.metrics_output_root}/{args.model_tag}/{date}"
        start_time_metrics = f"{date} {args.start_hour:02d}:00:00"

        run_step(
            [
                sys.executable,
                str(inference_py),
                "--start", start_time,
                "--end", end_time,
                "--x_path", args.x_path,
                "--checkpoint_path", args.checkpoint_path,
                "--device", args.device,
                "--save_path", inference_save_path,
                "--gfs_path", args.gfs_path,
            ],
            f"Inference {date}",
        )

        run_step(
            [
                sys.executable,
                str(replace_py),
                "--era5_path", args.era5_root,
                "--gfs_path", inference_save_path,
                "--output_root", replace_output_root,
                "--time_slice", date, date,
            ],
            f"Replace {date}",
        )

        replaced_zarr = f"{replace_output_root}/era5_localreplaced.zarr"
        if not Path(replaced_zarr).exists():
            fallback_zarr = "/home/ximutian/localreplaced/era5_localreplaced.zarr"
            if Path(fallback_zarr).exists():
                print(f"⚠️ 未找到 {replaced_zarr}，改用 {fallback_zarr}")
                replaced_zarr = fallback_zarr
            else:
                print(f"❌ 未找到替换结果，跳过后续步骤: {replaced_zarr}")
                continue

        run_step(
            [
                sys.executable,
                str(infer_onnx_py),
                "--input", replaced_zarr,
                "--save_dir", infer_onnx_save_dir,
                "--time_splite", date, date,
                "--device", args.device,
            ],
            f"Infer ONNX {date}",
        )

        run_step(
            [
                sys.executable,
                str(metrics_py),
                "--pred_root_era5", pred_root_era5,
                "--pred_root_naive_gfs", pred_root_naive_gfs,
                "--pred_root_trans_gfs", pred_root_trans_gfs,
                "--era5_root", args.era5_root,
                "--target_channel", args.target_channel,
                "--start_time", start_time_metrics,
                "--hour_interval", str(args.hour_interval),
                "--clim_path", args.clim_path,
                "--output_dir", metrics_output_dir,
                "--tag", args.model_tag,
                "--date_tag", date,
            ],
            f"Metrics {date}",
        )


if __name__ == "__main__":
    main()
