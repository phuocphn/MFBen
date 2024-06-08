```
python pinn_kd.py --multirun \
	scheme.network._target_=models.base.MLPConv \
	scheme.mode=train \
	scheme.network.hidden_layers=6 scheme.network.layer_neurons=64 \
	scheme.network.num_outputs=3 \
	scheme.mplsave_dir=plots/test/mlpconv.5k.h\${scheme.network.hidden_layers}.n\${scheme.network.layer_neurons}+distill.\${scheme.g_enable}T.\${scheme.T}.\${scheme.dataset_dir} \
	scheme.dataset_dir=sample_data/pygen/m.unit+case2+final/test/ellipse_set2_2,sample_data/pygen/m.unit+case2+final/test/equilateral_hexagon_2,sample_data/pygen/m.unit+case2+final/test/equilateral_octagon_2,sample_data/pygen/m.unit+case2+final/test/rectangle_set2_2,sample_data/pygen/m.unit+case2+final/test/semi_circle_2,sample_data/pygen/m.unit+case2+final/test/square_2,sample_data/pygen/m.unit+case2+final/test/trapezoid_2,sample_data/pygen/m.unit+case2+final/test/triangle_2 \
	scheme.save_pth_path=checkpoints/mlp.h\${scheme.network.hidden_layers}.n\${scheme.network.layer_neurons}+distill.\${scheme.g_enable}T.\${scheme.T}.pth \
	scheme.rho=1.0 \
	scheme.mu=0.01 \
	scheme.use_lrscheduler=true \
	scheme.training_scheme=pinn \
	scheme.load_data_mode=share \
	scheme.epochs=5000 \
	scheme.g_enable=true \
	scheme.g_pretrained=pretrained_models/g_teacher/pointnetcfd.pipn-1.5k.pth \
	scheme.T=1,2,5,10,32
```
