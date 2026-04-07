"""
Deploy the Color Calibration demo to HuggingFace Spaces.

Usage:
    python scripts/deploy_space.py --space_id jeffliulab/color_calibration

Prerequisites:
    - pip install huggingface_hub
    - HF token in ~/.hf_token or HF_TOKEN env var
    - git-lfs installed (for checkpoint upload)

What this script does:
    1. Creates the HF Space repo (if it doesn't exist)
    2. Clones it to a temp directory
    3. Copies all files from space/ into the clone
    4. Configures git-lfs for .pt and .pkl files
    5. Commits and pushes
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE_DIR = ROOT / "space"


def get_token():
    token_path = Path.home() / ".hf_token"
    if token_path.exists():
        return token_path.read_text().strip()
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    print("ERROR: No HuggingFace token found.")
    print("  Option 1: echo 'hf_xxx' > ~/.hf_token")
    print("  Option 2: export HF_TOKEN=hf_xxx")
    sys.exit(1)


def run(cmd, cwd=None):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr.strip()}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Deploy to HuggingFace Spaces")
    parser.add_argument("--space_id", type=str, default="jeffliulab/color_calibration",
                        help="HF Space ID, e.g. jeffliulab/color_calibration")
    parser.add_argument("--message", type=str, default="Update Space",
                        help="Commit message")
    args = parser.parse_args()

    token = get_token()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=token)

    # 1. Create Space if it doesn't exist
    print(f"\n[1/4] Ensuring Space exists: {args.space_id}")
    try:
        api.create_repo(
            repo_id=args.space_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
        )
        print(f"  Space ready: https://huggingface.co/spaces/{args.space_id}")
    except Exception as e:
        print(f"  Failed to create Space: {e}")
        sys.exit(1)

    # 2. Clone
    clone_dir = ROOT / "_hf_space_deploy"
    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    print(f"\n[2/4] Cloning Space repo...")
    clone_url = f"https://USER:{token}@huggingface.co/spaces/{args.space_id}"
    result = run(f'git clone "{clone_url}" "{clone_dir}"')
    if result.returncode != 0:
        clone_dir.mkdir()
        run("git init", cwd=clone_dir)
        run(f'git remote add origin "https://USER:{token}@huggingface.co/spaces/{args.space_id}"',
            cwd=clone_dir)

    # 3. Copy files from space/
    print(f"\n[3/4] Copying files from space/ ...")

    # Clear old files (except .git)
    for item in clone_dir.iterdir():
        if item.name == ".git":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    # Copy everything from space/
    for item in SPACE_DIR.iterdir():
        dst = clone_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    print(f"  Copied {sum(1 for _ in clone_dir.rglob('*') if _.is_file())} files")

    # 4. Git LFS + commit + push
    print(f"\n[4/4] Committing and pushing...")
    run("git lfs install", cwd=clone_dir)
    run('git lfs track "*.pt"', cwd=clone_dir)
    run('git lfs track "*.pkl"', cwd=clone_dir)
    run("git add -A", cwd=clone_dir)
    run(f'git commit -m "{args.message}"', cwd=clone_dir)
    run("git push origin main", cwd=clone_dir)

    # Cleanup
    shutil.rmtree(clone_dir, ignore_errors=True)

    print(f"\nDone! View your Space at:")
    print(f"  https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
