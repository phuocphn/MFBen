import os
import itertools
from pathlib import Path
import subprocess
import sys
import time

lr_scheduler_enable = True
epochs = 50_000

SAVE_DIR = os.path.join(
    "experiment-data",
    f"pipn",
)
CHECK_POINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
CHECK_POINT_PATH = os.path.join(CHECK_POINT_DIR, "checkpoint.pth")
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
Path(CHECK_POINT_DIR).mkdir(parents=True, exist_ok=True)

command = [
    "python",
    "pinn_kd.py",
    "--multirun",
    "scheme.network._target_=models.base.PointNetSeg",
    "~scheme.network.hidden_layers",
    "~scheme.network.layer_neurons",
    "~scheme.network.num_outputs",
    f"+scheme.network.use_max_fn={False}",
    f"+scheme.network.use_bn={False}",
    "scheme.mode=train",
    # "scheme.network.num_outputs=3",
    f"scheme.mplsave_dir={SAVE_DIR}",
    "scheme.dataset_dir=datasets/pipn.1/",
    f"scheme.save_pth_path={CHECK_POINT_PATH}",
    "scheme.rho=1.0",
    "scheme.mu=0.01",
    f"scheme.lr_scheduler_enable={lr_scheduler_enable}",
    "scheme.training_scheme=pinn",
    "scheme.load_data_mode=normal",  # main difference
    "scheme.batch_size=32",  # main difference
    f"scheme.epochs={epochs}",
    f"scheme.g_enable={False}",
]

os.system(" ".join(command))
