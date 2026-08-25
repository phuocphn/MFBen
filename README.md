# MFBen

Physics-informed neural networks (PINNs) for steady 2D incompressible flow over
varying geometries, and knowledge distillation from a large PointNet-style
teacher into small MLPConv students.

The network takes a point cloud of a geometry and predicts the velocity and
pressure fields `(u, v, p)` at every point. Training is driven by the
Navier–Stokes residual rather than by labels, so a model can be fit to a new
geometry without a CFD solution for it — the reference OpenFOAM fields are used
only for evaluation.

## What is in here

| Path | Purpose |
| --- | --- |
| `pinn_kd.py` | Main training/eval entry point (Hydra app). |
| `conf/` | Hydra config: `conf/config.yaml` selects `conf/scheme/pinn.yaml`. |
| `models/` | `PointNetSeg`, `MLPConv`, `PointNetCFD`, MLP and ResNet baselines. |
| `pinnstorch/` | Point-cloud domains, CFD dataloaders, autograd gradient helpers. |
| `dsgen/` | Dataset generation: gmsh `.geo` geometries → OpenFOAM → `.pth` tensors. |
| `scripts/` | Sweep launchers that shell out to `pinn_kd.py --multirun`. |
| `post-processing/` | Aggregates result JSONs into the summary spreadsheets. |
| `docs/` | `training_notes.md`, `submission_1.md` with the exact commands used. |
| `experiment-data*/`, `results/` | Committed metrics — see [Results](#results). |

## Physics and losses

`calc_pde_loss` in `pinn_kd.py` builds the residuals of the steady 2D
incompressible Navier–Stokes momentum equations plus continuity, with density
`rho` and viscosity `mu` (defaults `1.0` and `0.01`). Derivatives come from
`torch.autograd`, not finite differences.

Without distillation the objective is simply

```
loss = pde_loss + bc_loss
```

With `scheme.g_enable=true`, a frozen teacher `g` supplies soft targets and the
student adds a temperature-scaled KL term:

```
distill_loss = KL(softmax(teacher/T) || log_softmax(student/T)) * T^2
loss         = lambda1 * (pde_loss + bc_loss) + lambda2 * distill_loss
```

`lambda1` ramps 0→1 and `lambda2` ramps 1→0 linearly over the first 2000
epochs, after which training is pure physics. So distillation acts as a warm
start that hands off to the PDE residual, rather than a term that competes with
it for the whole run.

## Setup

Training needs PyTorch, Hydra (`hydra-core`, `omegaconf`), `wandb`, `tqdm`,
`numpy`, `matplotlib` and `pandas`. Dataset generation additionally needs an
OpenFOAM installation on `PATH` (for `simpleFoam`), plus `pyvista`, `vtk`,
`h5py` and `click`. A CUDA device is used when available, otherwise CPU.

## Generating the dataset

Geometries live in `dsgen/geo/` — 18 training and 8 test shapes under

heart, house, quatrefoil, star) under `extra/`. The pipeline is three ordered
steps run from `dsgen/`:

```bash
python 01_cp_templete_dir.py   # clone the OpenFOAM case template per geometry
python 02_openfoam_sim.py      # run simpleFoam to steady state
python 03_write_pytorch_pth.py # sample 5000 cells, write Cx/Cy/Ux/Uy/p tensors
```

Each step reads a `GEO_DIR` set at the top of the file, so point it at the
geometry set you want before running.

## Training

Single run — everything is a Hydra override under `scheme.`:

```bash
python pinn_kd.py \
  scheme.mode=train \
  scheme.network._target_=models.base.MLPConv \
  scheme.network.hidden_layers=6 scheme.network.layer_neurons=64 \
  scheme.dataset_dir=sample_data/pygen/m.unit+case2+final/test/square_2 \
  scheme.epochs=5000 \
  scheme.g_enable=false
```

The sweep launchers reproduce the published grids:

```bash
python scripts/non-kd.py   # MLPConv baselines, no distillation
python scripts/kd.py       # same grid, distilled at T = 1, 2, 5, 10, 32
python scripts/pipn.py     # PointNetSeg / PIPN teacher, 50k epochs
```

`scripts/kd.py` and `scripts/non-kd.py` sweep hidden layers `[3, 6, 10]` ×
neurons `[32, 64, 128]` over the 8 test geometries at 5000 epochs. The
`*_extra.py` variants run the same thing on the `extra/` shapes. Distillation
loads its frozen teacher from `scheme.g_pretrained`, which defaults to the
`pretrained_models/g_teacher/pointnetcfd.pipn-1.5k.pth` checkpoint committed
here (PointNetSeg, 866,947 params, saved at epoch 4900), so `scripts/kd.py`
runs straight from a fresh clone.

Checkpoints go to `checkpoints/`, plots and per-epoch CSV logs to the directory
given by `scheme.mplsave_dir`.

## Results

Model weights are far too large for git and are not tracked. What *is*
committed is everything needed to rebuild the tables and curves:

- `experiment-data/` and `experiment-data-pipn-#3/` — per-geometry result JSONs
  for the `kd/<T>/`, `non-kd/` and `pipn/` runs. Each holds `l2_u_error`,
  `l2_v_error`, `l2_p_error` and `mse_error` lists plus their `avg.*` scalars.
- `results/experiment-data-#1/`, `results/experiment-data-#2/` — the two earlier
  experiment rounds, same layout.
- `results/data-#1.xlsx`, `results/data-#2.xlsx` — the aggregated summaries.
- `results/MLCAD_submission/plots/**/training_logs.csv` — per-epoch curves
  (`epoch, u, v, p, val_loss, total_loss, pde_loss, collocation_loss, bc_loss`)
  for the 50k-epoch baseline and the 8 distilled test geometries.

To regenerate a spreadsheet from the JSONs:

```bash
python post-processing/get_l2_improvements.py
```

It averages `avg.u/v/p` across geometries per network configuration, then across
the `h{3,6,10}.n{32,64,128}` grid and `pointnetcfd` where present.

## Notes

`.gitignore` deliberately excludes `checkpoints/`, `plots*`, `outputs/`,
`multirun/`, `datasets/`, `sample_data/` and `wandb/`. Two exceptions are
tracked on purpose:

- `results/MLCAD_submission/plots/**/training_logs.csv` — force-added past the
  `plots*` rule, since they are the only record of the loss curves.
- `pretrained_models/g_teacher/` — admitted by an explicit `!` rule so the KD
  sweep is runnable out of the box. The other checkpoints under
  `pretrained_models/` remain ignored.
