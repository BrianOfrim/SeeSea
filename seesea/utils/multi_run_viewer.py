"""
In order to view multiple runs at the same time we need to format a command that
we can pass to tensorboard's logdir_spec argument.

Example input structure:
/root/2024_12_20_1200/runs
/root/2024_12_21_1110/runs
/root/2024_12_22_0200/runs

Example command that we want to run:
tensorboard --logdir_spec "2024_12_20_1200:/root/2024_12_20_1200/runs,2024_12_21_1110:/root/2024_12_21_1110/runs,2024_12_22_0200:/root/2024_12_22_0200/runs"

"""

import os
import argparse
import subprocess


def format_logdir_spec(root_dir):
    """Format the logdir_spec string for tensorboard from a directory of runs.

    Args:
        root_dir: Path to directory containing run folders

    Returns:
        Formatted logdir_spec string for tensorboard
    """
    # Get all subdirectories that contain a 'run' folder
    run_dirs = []
    for item in os.listdir(root_dir):
        full_path = os.path.join(root_dir, item)
        if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "runs")):
            run_dirs.append(f"{item}:{os.path.join(root_dir, item)}")

    # Join with commas
    logdir_spec = ",".join(run_dirs)

    return logdir_spec


def main(args):
    logdir_spec = format_logdir_spec(args.root_dir)

    print(logdir_spec)

    # Construct and run tensorboard command
    cmd = ["tensorboard", "--logdir_spec", logdir_spec]

    print(cmd)
    subprocess.run(cmd, cwd=args.root_dir, check=True)


def get_args_parser():
    parser = argparse.ArgumentParser(description="View multiple tensorboard runs simultaneously")
    parser.add_argument("--root_dir", help="Root directory containing training runs")
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
