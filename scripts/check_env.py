import importlib
import sys

import torch


REQUIRED = [
    "PIL",
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "yaml",
    "tqdm",
    "scipy",
    "huggingface_hub",
]


def main():
    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"torch cuda build: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x.T
        print(f"cuda matmul ok: {float(y.mean().detach().cpu()):.6f}")
    missing = []
    for name in REQUIRED:
        try:
            importlib.import_module(name)
        except Exception as exc:
            missing.append((name, str(exc)))
    if missing:
        print("missing dependencies:")
        for name, exc in missing:
            print(f"- {name}: {exc}")
        raise SystemExit(1)
    print("all project dependencies ok")


if __name__ == "__main__":
    main()
