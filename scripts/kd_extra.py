import os
import itertools
from pathlib import Path
import subprocess
import sys
import time

# Baseline, Knowledge Distillation: Disable
# [3,6,10], [32,64,128],[1,2,5,10,32]
for hidden_layers, layer_neurons, temperature in list(
    itertools.product([3, 6, 10], [32, 64, 128], [1, 2, 5, 10, 32])
):
    # hidden_layers = 6
    # layer_neurons = 64
    lr_scheduler_enable = True
    epochs = 5000

    SAVE_DIR = os.path.join(
        "experiment-data",
        f"kd",
        str(temperature),
        f"mlpconv.h{hidden_layers}.n{layer_neurons}",
    )
    CHECK_POINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
    CHECK_POINT_PATH = os.path.join(CHECK_POINT_DIR, "checkpoint.pth")
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECK_POINT_DIR).mkdir(parents=True, exist_ok=True)

    command = [
        "python",
        "pinn_kd.py",
        "--multirun",
        "scheme.network._target_=models.base.MLPConv",
        "scheme.mode=train",
        f"scheme.network.hidden_layers={hidden_layers} scheme.network.layer_neurons={layer_neurons}",
        "scheme.network.num_outputs=3",
        f"scheme.mplsave_dir={SAVE_DIR}",
        "scheme.dataset_dir=datasets/extra/crescent,datasets/extra/heart,datasets/extra/house,datasets/extra/quatrefoil,datasets/extra/star",
        f"scheme.save_pth_path={CHECK_POINT_PATH}",
        "scheme.rho=1.0",
        "scheme.mu=0.01",
        f"scheme.lr_scheduler_enable={lr_scheduler_enable}",
        "scheme.training_scheme=pinn",
        "scheme.load_data_mode=share",
        f"scheme.epochs={epochs}",
        "scheme.g_enable=true",
        "scheme.g_pretrained=pretrained_models/g_teacher/pointnetcfd.pipn-1.5k.pth",
        f"scheme.T={temperature}",
    ]

    os.system(" ".join(command))
    time.sleep(60)
