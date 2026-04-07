import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd, title):
    print(f"\n=== {title} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Batch inference for naive GFS")
    parser.add_argument(
        "--dates",
        type=str,
        nargs="+",
        default=["20250601", "20250615"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--input_root", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/localreplaced/with_naive_gfs_real")
    parser.add_argument("--save_dir", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/inference_naive_gfs")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--hour_interval", type=int, default=6)
    parser.add_argument("--init_time_hour", type=str, nargs="+", default=["0", "12"])
    parser.add_argument("--total_step", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "fp32"])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    infer_onnx_py = repo_root / "fuxi-rtm" / "fuxi-rtm" / "infer_onnx_release.py"

    for date in args.dates:
        input_path = f"{args.input_root}/{date}/era5_localreplaced.zarr"
        if not Path(input_path).exists():
            print(f"\n=== Naive GFS Infer {date} ===")
            print(f"跳过：输入不存在 {input_path}")
            continue

        cmd = [
            sys.executable,
            str(infer_onnx_py),
            "--input", input_path,
            "--save_dir", args.save_dir,
            "--time_splite", date, date,
            "--device", args.device,
            "--hour_interval", str(args.hour_interval),
            "--total_step", str(args.total_step),
            "--batch_size", str(args.batch_size),
            "--dtype", args.dtype,
            "--init_time_hour", *args.init_time_hour,
        ]
        if args.model:
            cmd.extend(["--model", args.model])

        run_step(cmd, f"Naive GFS Infer {date}")


if __name__ == "__main__":
    main()
