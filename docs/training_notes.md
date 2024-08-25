## Training PINN with Knowledge Distillation 🛫

For training MLPConv models, simply run the bash scripts in `scripts/kd.py` (for PINN+KD) and `scripts/non-kd.py` (for PINN only).

For training PointNet CFD models, run the following scripts.

**Non-Knowledge Distillation**

```bash
python pinn_kd.py --multirun scheme.network._target_=models.base.PointNetSeg scheme.mode=train ~scheme.network.hidden_layers ~scheme.network.layer_neurons ~scheme.network.num_outputs +scheme.network.use_max_fn=false +scheme.network.use_bn=false scheme.mplsave_dir=experiment-data/non-kd/pointnetcfd/   scheme.dataset_dir=sample_data/pygen/m.unit+case2+final/test/ellipse_set2_2,sample_data/pygen/m.unit+case2+final/test/equilateral_hexagon_2,sample_data/pygen/m.unit+case2+final/test/equilateral_octagon_2,sample_data/pygen/m.unit+case2+final/test/rectangle_set2_2,sample_data/pygen/m.unit+case2+final/test/semi_circle_2,sample_data/pygen/m.unit+case2+final/test/square_2,sample_data/pygen/m.unit+case2+final/test/trapezoid_2,sample_data/pygen/m.unit+case2+final/test/triangle_2 scheme.rho=1.0 scheme.mu=0.01 scheme.use_lrscheduler=true scheme.training_scheme=pinn scheme.load_data_mode=share scheme.epochs=5000 scheme.g_enable=false
```

**Knowledge Distillation**

```bash
python pinn_kd.py --multirun scheme.network._target_=models.base.PointNetSeg scheme.mode=train ~scheme.network.hidden_layers ~scheme.network.layer_neurons ~scheme.network.num_outputs +scheme.network.use_max_fn=false +scheme.network.use_bn=false scheme.mplsave_dir=experiment-data/kd/\${scheme.T}/pointnetcfd/   scheme.dataset_dir=sample_data/pygen/m.unit+case2+final/test/ellipse_set2_2,sample_data/pygen/m.unit+case2+final/test/equilateral_hexagon_2,sample_data/pygen/m.unit+case2+final/test/equilateral_octagon_2,sample_data/pygen/m.unit+case2+final/test/rectangle_set2_2,sample_data/pygen/m.unit+case2+final/test/semi_circle_2,sample_data/pygen/m.unit+case2+final/test/square_2,sample_data/pygen/m.unit+case2+final/test/trapezoid_2,sample_data/pygen/m.unit+case2+final/test/triangle_2 scheme.rho=1.0 scheme.mu=0.01 scheme.use_lrscheduler=true scheme.training_scheme=pinn scheme.load_data_mode=share scheme.epochs=5000 scheme.g_enable=true scheme.T=1,2,5,10,32
```

After that, use the script in `post-processing/get_l2_improvements.py` to obtain the final result in the generated `data.xlsx` file.
