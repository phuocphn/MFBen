from __future__ import print_function

import torch
import numpy as np
import random
import glob

import wandb
import hydra
import os
import json
import csv

from tqdm import tqdm
from pathlib import Path
from models import mlp
from models import pointnetcfd
from models import mlp_conv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.autograd as autograd
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from torch.linalg import norm
from pathlib import Path

from pinnstorch.data.domains.point_cloud import PointData
from pinnstorch.data.dataloader import GeometryDataLoader, CFDDomain, PDAcrossCFDDomain

# from pinnstorch.data import GeometryDataLoader
from pinnstorch.data import create_cfd_domain, create_cfddomain_set
from omegaconf import DictConfig, OmegaConf
from visualize import visualize, visualize_single, plot_solution

device = "cuda" if torch.cuda.is_available() else "cpu"
# ensure reproducibility (https://pytorch.org/docs/stable/notes/randomness.html)
seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

from dataclasses import dataclass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def to_point_data(pt_file) -> PointData:
#     data = torch.load(pt_file)
#     assert data.shape[0] == 5
#     assert data.shape[1] > 100
#     x = data[0, :].view(1, 1, -1).to(device)
#     y = data[1, :].view(1, 1, -1).to(device)

#     u = data[2, :].view(1, 1, -1).to(device)
#     v = data[3, :].view(1, 1, -1).to(device)
#     p = data[4, :].view(1, 1, -1).to(device)
#     return PointData(x, y, u, v, p)


def gradient(y, x):
    return autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def pde_loss(x, y, preds, extra_variables, rho=1.0, mu=0.01):
    assert preds.shape[-2] == 3
    u, v, p = preds[:, 0:1, :], preds[:, 1:2, :], preds[:, 2:3, :]

    p_x = gradient(p, x).reshape([-1])
    p_y = gradient(p, y).reshape([-1])

    u_x = gradient(u, x).reshape([-1])
    u_y = gradient(u, y).reshape([-1])
    u_xx = gradient(gradient(u, x), x).reshape([-1])
    u_yy = gradient(gradient(u, y), y).reshape([-1])

    v_x = gradient(v, x).reshape([-1])
    v_y = gradient(v, y).reshape([-1])
    v_xx = gradient(gradient(v, x), x).reshape([-1])
    v_yy = gradient(gradient(v, y), y).reshape([-1])

    f_u = (
        rho * (u.reshape([-1]) * u_x + v.reshape([-1]) * u_y) + p_x - mu * (u_xx + u_yy)
    )
    f_v = (
        rho * (u.reshape([-1]) * v_x + v.reshape([-1]) * v_y) + p_y - mu * (v_xx + v_yy)
    )

    f_mass = u_x + v_y
    loss_f_u = F.mse_loss(f_u, torch.zeros_like(f_u))
    loss_f_v = F.mse_loss(f_v, torch.zeros_like(f_v))
    loss_mass = F.mse_loss(f_mass, torch.zeros_like(f_mass))
    loss = loss_f_u + loss_f_v + loss_mass
    return loss


from collections import namedtuple

SampledData = namedtuple("SampledData", ["x", "y", "u", "v", "p"])


def sample_data(org: PointData, k=4096) -> SampledData:
    li = torch.tensor(random.sample(range(org.x.shape[-1]), k)).cuda()
    x = org.x.view(-1)[li]
    y = org.y.view(-1)[li]
    u = org.u.view(-1)[li]
    v = org.v.view(-1)[li]
    p = org.p.view(-1)[li]
    return SampledData(x, y, u, v, p)


def construct_io(x_b, y_b, u_b, v_b, p_b):
    inputs_x = torch.tensor(torch.stack(x_b), dtype=torch.float32, requires_grad=True)
    inputs_y = torch.tensor(torch.stack(y_b), dtype=torch.float32, requires_grad=True)
    _inputs = torch.cat(
        [inputs_x.unsqueeze(1), inputs_y.unsqueeze(1)], dim=1
    ).requires_grad_(True)

    _outputs = torch.cat(
        [
            torch.stack(u_b).unsqueeze(1),
            torch.stack(v_b).unsqueeze(1),
            torch.stack(p_b).unsqueeze(1),
        ],
        dim=1,
    ).requires_grad_(False)
    return _inputs, _outputs, inputs_x, inputs_y


def train_batch(
    coll_points_raw,
    boundary_points_raw,
    inlets_raw,
    outlets_raw,
    obstacles_raw,
    walls_raw,
    model,
    optimizer,
    lr_scheduler,
    criterion,
    ni,
    extra_variables,
    config,
):

    model.train()
    optimizer.zero_grad()

    coll = {"x": [], "y": [], "u": [], "v": [], "p": []}
    bc = {"x": [], "y": [], "u": [], "v": [], "p": []}
    bc_inlets = {"x": [], "y": [], "u": [], "v": [], "p": []}
    bc_outlets = {"x": [], "y": [], "u": [], "v": [], "p": []}
    bc_obstacles = {"x": [], "y": [], "u": [], "v": [], "p": []}
    bc_walls = {"x": [], "y": [], "u": [], "v": [], "p": []}

    infox = {"obstacle": 262, "wall": 404, "inlet": 202, "outlet": 202}

    for coll_points, boundary_points, inlets, outlets, obstacles, walls in zip(
        coll_points_raw,
        boundary_points_raw,
        inlets_raw,
        outlets_raw,
        obstacles_raw,
        walls_raw,
    ):
        data = sample_data(coll_points, k=25_000)
        for physical_field in ["x", "y", "u", "v", "p"]:
            coll[physical_field].append(getattr(data, physical_field))

        data = sample_data(boundary_points, k=600)
        for physical_field in ["x", "y", "u", "v", "p"]:
            bc[physical_field].append(getattr(data, physical_field))

        data = sample_data(inlets, k=infox["inlet"])
        for physical_field in ["x", "y", "u", "v", "p"]:
            bc_inlets[physical_field].append(getattr(data, physical_field))

        data = sample_data(outlets, k=infox["outlet"])
        for physical_field in ["x", "y", "u", "v", "p"]:
            bc_outlets[physical_field].append(getattr(data, physical_field))

        data = sample_data(obstacles, k=infox["obstacle"])
        for physical_field in ["x", "y", "u", "v", "p"]:
            bc_obstacles[physical_field].append(getattr(data, physical_field))

        data = sample_data(walls, k=infox["wall"])
        for physical_field in ["x", "y", "u", "v", "p"]:
            bc_walls[physical_field].append(getattr(data, physical_field))

    coll_inputs, coll_outputs, coll_x, coll_y = construct_io(
        coll["x"], coll["y"], coll["u"], coll["v"], coll["p"]
    )
    bc_inputs, bc_outputs, _, _ = construct_io(
        bc["x"], bc["y"], bc["u"], bc["v"], bc["p"]
    )
    bc_inlet_inputs, bc_inlet_outputs, _, _ = construct_io(
        bc_inlets["x"], bc_inlets["y"], bc_inlets["u"], bc_inlets["v"], bc_inlets["p"]
    )
    bc_outlet_inputs, bc_outlet_outputs, _, _ = construct_io(
        bc_outlets["x"],
        bc_outlets["y"],
        bc_outlets["u"],
        bc_outlets["v"],
        bc_outlets["p"],
    )
    bc_obstacle_inputs, bc_obstacle_outputs, _, _ = construct_io(
        bc_obstacles["x"],
        bc_obstacles["y"],
        bc_obstacles["u"],
        bc_obstacles["v"],
        bc_obstacles["p"],
    )
    bc_wall_inputs, bc_wall_outputs, _, _ = construct_io(
        bc_walls["x"],
        bc_walls["y"],
        bc_walls["u"],
        bc_walls["v"],
        bc_walls["p"],
    )

    preds = model(coll_inputs)
    if config.training_scheme == "data":
        loss = F.mse_loss(preds, coll_outputs) + F.mse_loss(
            model(bc_inputs), bc_outputs
        )
        bc_loss = _pde_loss = collocation_loss = 0

    elif config.training_scheme == "pinn":

        _pde_loss = pde_loss(
            coll_x, coll_y, preds, extra_variables, rho=config.rho, mu=config.mu
        )
        bc_loss = (
            F.mse_loss(model(bc_wall_inputs)[:, :2, :], bc_wall_outputs[:, :2, :])
            + F.mse_loss(
                model(bc_obstacle_inputs)[:, :2, :], bc_obstacle_outputs[:, :2, :]
            )
            + F.mse_loss(model(bc_inlet_inputs)[:, :1, :], bc_inlet_outputs[:, :1, :])
            + F.mse_loss(
                model(bc_outlet_inputs)[:, 2:3, :], bc_outlet_outputs[:, 2:3, :]
            )
        )
        collocation_loss = F.mse_loss(model(coll_inputs), coll_outputs)
        loss = _pde_loss + bc_loss

    loss.backward()
    optimizer.step()

    if config.use_lrscheduler:
        lr_scheduler.step()
    return {
        "total_loss": loss,
        "bc_loss": bc_loss,
        "pde_loss": _pde_loss,
        "collocation_loss": collocation_loss,
    }


def train_batch_pinn_single(
    cfddata: PDAcrossCFDDomain,
    model,
    optimizer,
    lr_scheduler,
    criterion,
    ni,
    extra_variables,
    config,
    g,
    epoch,
):
    def _custom_pde_loss(x, y, preds, extra_variables, rho, mu):
        assert preds.shape[-2] == 3

        u, v, p = preds[:, 0:1, :], preds[:, 1:2, :], preds[:, 2:3, :]

        p_x = autograd.grad(p, x, torch.ones_like(x), create_graph=True)[0]
        p_y = autograd.grad(p, y, torch.ones_like(y), create_graph=True)[0]

        u_x = autograd.grad(u, x, torch.ones_like(x), create_graph=True)[0]
        u_y = autograd.grad(u, y, torch.ones_like(y), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x, torch.ones_like(x), create_graph=True)[0]
        u_yy = autograd.grad(u_y, y, torch.ones_like(y), create_graph=True)[0]

        v_x = autograd.grad(v, x, torch.ones_like(x), create_graph=True)[0]
        v_y = autograd.grad(v, y, torch.ones_like(y), create_graph=True)[0]
        v_xx = autograd.grad(v_x, x, torch.ones_like(x), create_graph=True)[0]
        v_yy = autograd.grad(v_y, y, torch.ones_like(y), create_graph=True)[0]

        f_u = rho * (u * u_x + v * u_y) + p_x - mu * (u_xx + u_yy)
        f_v = rho * (u * v_x + v * v_y) + p_y - mu * (v_xx + v_yy)
        f_mass = u_x + v_y

        loss_f_u = F.mse_loss(f_u, torch.zeros_like(f_u))
        loss_f_v = F.mse_loss(f_v, torch.zeros_like(f_v))
        loss_mass = F.mse_loss(f_mass, torch.zeros_like(f_mass))
        loss = loss_f_u + loss_f_v + loss_mass
        return loss

    model.train()
    optimizer.zero_grad()

    # only for single_batch training (adding 0 index)
    coll_points = cfddata.dd_collocation[0]
    boundary_points = cfddata.dd_boundary[0]

    coll_inputs, raw_x, raw_y, indices1 = coll_points.construct_cloud_inputs(k=2048)
    coll_targets = coll_points.construct_cloud_targets(indices1)
    bc_inputs, _, _ = boundary_points.construct_inputs()
    bc_targets = boundary_points.construct_targets()

    preds = model(coll_inputs)
    pde_loss = _custom_pde_loss(
        raw_x, raw_y, preds, extra_variables, rho=config.rho, mu=config.mu
    )

    coll_loss = F.mse_loss(model(coll_inputs), coll_targets)
    bc_loss = F.mse_loss(model(bc_inputs), bc_targets)

    NUM_COLL_SAMPLES_FOR_KD = 20_000
    if config.g_enable:
        kl_loss = nn.KLDivLoss()

        dd_inputs = coll_points.construct_cloud_inputs(k=NUM_COLL_SAMPLES_FOR_KD)[0]
        with torch.no_grad():
            teacher_logits = g(dd_inputs)

        batch_size = teacher_logits.shape[0]
        student_logits = model(dd_inputs)
        soft_targets = F.softmax(teacher_logits.view(batch_size, -1) / config.T, dim=-1)
        soft_prob = F.log_softmax(
            student_logits.view(batch_size, -1) / config.T, dim=-1
        )

        distill_loss = (
            torch.sum(soft_targets * (soft_targets.log() - soft_prob))
            / soft_prob.size()[0]
            * (config.T**2)
        )

        if epoch < 2000:
            lambda1 = np.linspace(0, 1, 2000)[epoch]
            lambda2 = np.linspace(1, 0, 2000)[epoch]
        else:
            lambda1 = 1
            lambda2 = 0
        if epoch == 0:
            print("distill_loss:", distill_loss, "dd_inputs:", dd_inputs.shape[-1])

        loss = lambda1 * (pde_loss + bc_loss) + lambda2 * distill_loss
    else:
        loss = pde_loss + bc_loss

    loss.backward()
    optimizer.step()
    if config.lr_scheduler_enable:
        lr_scheduler.step()

    return {
        "total_loss": loss,
        "bc_loss": bc_loss,
        "pde_loss": pde_loss,
        "collocation_loss": coll_loss,
    }


def train_batch_pinn_multiple(
    coll_points,
    boundary_points,
    inlets_raw,
    outlets_raw,
    obstacles_raw,
    walls_raw,
    model,
    optimizer,
    lr_scheduler,
    criterion,
    ni,
    extra_variables,
    config,
):
    def _custom_pde_loss(x, y, preds, extra_variables, rho, mu):
        assert preds.shape[-2] == 3

        u, v, p = preds[:, 0:1, :], preds[:, 1:2, :], preds[:, 2:3, :]

        p_x = autograd.grad(p, x, torch.ones_like(x), create_graph=True)[0]
        p_y = autograd.grad(p, y, torch.ones_like(y), create_graph=True)[0]

        u_x = autograd.grad(u, x, torch.ones_like(x), create_graph=True)[0]
        u_y = autograd.grad(u, y, torch.ones_like(y), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x, torch.ones_like(x), create_graph=True)[0]
        u_yy = autograd.grad(u_y, y, torch.ones_like(y), create_graph=True)[0]

        v_x = autograd.grad(v, x, torch.ones_like(x), create_graph=True)[0]
        v_y = autograd.grad(v, y, torch.ones_like(y), create_graph=True)[0]
        v_xx = autograd.grad(v_x, x, torch.ones_like(x), create_graph=True)[0]
        v_yy = autograd.grad(v_y, y, torch.ones_like(y), create_graph=True)[0]

        f_u = rho * (u * u_x + v * u_y) + p_x - mu * (u_xx + u_yy)
        f_v = rho * (u * v_x + v * v_y) + p_y - mu * (v_xx + v_yy)
        f_mass = u_x + v_y

        loss_f_u = F.mse_loss(f_u, torch.zeros_like(f_u))
        loss_f_v = F.mse_loss(f_v, torch.zeros_like(f_v))
        loss_mass = F.mse_loss(f_mass, torch.zeros_like(f_mass))
        loss = loss_f_u + loss_f_v + loss_mass
        return loss

    model.train()
    optimizer.zero_grad()

    for coll, bc in zip(coll_points, boundary_points):
        inputs, raw_x, raw_y, indices1 = coll.construct_cloud_inputs(k=2048)
        preds = model(inputs)
        import pudb

        _pde_loss = _custom_pde_loss(
            raw_x, raw_y, preds, extra_variables, rho=config.rho, mu=config.mu
        )
        _bc_inputs, _, _ = bc.construct_inputs()
        _bc_outputs = bc.construct_targets()
        bc_loss = F.mse_loss(
            model(_bc_inputs),
            _bc_outputs,
        )
        collocation_loss = F.mse_loss(
            model(inputs), coll.construct_cloud_targets(indices1)
        )
        loss = _pde_loss + bc_loss
        loss.backward()
    optimizer.step()
    if config.use_lrscheduler:
        lr_scheduler.step()
    return {
        "total_loss": loss,
        "bc_loss": bc_loss,
        "pde_loss": _pde_loss,
        "collocation_loss": collocation_loss,
    }


def train_batch_pinn_multiplev2(
    coll_points,
    boundary_points,
    inlets_raw,
    outlets_raw,
    obstacles_raw,
    walls_raw,
    model,
    optimizer,
    lr_scheduler,
    criterion,
    ni,
    extra_variables,
    config,
):
    def _custom_pde_loss(x, y, preds, extra_variables, rho, mu):
        assert preds.shape[-2] == 3

        u, v, p = preds[:, 0:1, :], preds[:, 1:2, :], preds[:, 2:3, :]

        p_x = autograd.grad(p, x, torch.ones_like(x), create_graph=True)[0]
        p_y = autograd.grad(p, y, torch.ones_like(y), create_graph=True)[0]

        u_x = autograd.grad(u, x, torch.ones_like(x), create_graph=True)[0]
        u_y = autograd.grad(u, y, torch.ones_like(y), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x, torch.ones_like(x), create_graph=True)[0]
        u_yy = autograd.grad(u_y, y, torch.ones_like(y), create_graph=True)[0]

        v_x = autograd.grad(v, x, torch.ones_like(x), create_graph=True)[0]
        v_y = autograd.grad(v, y, torch.ones_like(y), create_graph=True)[0]
        v_xx = autograd.grad(v_x, x, torch.ones_like(x), create_graph=True)[0]
        v_yy = autograd.grad(v_y, y, torch.ones_like(y), create_graph=True)[0]

        f_u = rho * (u * u_x + v * u_y) + p_x - mu * (u_xx + u_yy)
        f_v = rho * (u * v_x + v * v_y) + p_y - mu * (v_xx + v_yy)
        f_mass = u_x + v_y

        loss_f_u = F.mse_loss(f_u, torch.zeros_like(f_u))
        loss_f_v = F.mse_loss(f_v, torch.zeros_like(f_v))
        loss_mass = F.mse_loss(f_mass, torch.zeros_like(f_mass))
        # loss = loss_f_u + loss_f_v + loss_mass
        # return loss
        return loss_f_u, loss_f_v, loss_mass

    model.train()
    optimizer.zero_grad()

    ff_u = []
    ff_v = []
    ff_continuty = []
    bbc = []
    for coll, bc in zip(coll_points, boundary_points):
        # coll_points = coll_points[0]
        # boundary_points = boundary_points[0]
        inputs, raw_x, raw_y, indices1 = coll.construct_cloud_inputs(k=2048)
        preds = model(inputs)
        import pudb

        loss_f_u, loss_f_v, loss_mass = _custom_pde_loss(
            raw_x, raw_y, preds, extra_variables, rho=config.rho, mu=config.mu
        )
        ff_u.append(loss_f_u)
        ff_v.append(loss_f_v)
        ff_continuty.append(loss_mass)

        _bc_inputs, _, _ = bc.construct_inputs()
        _bc_outputs = bc.construct_targets()
        bc_loss = F.mse_loss(
            model(_bc_inputs),
            _bc_outputs,
        )
        bbc.append(bc_loss)
        collocation_loss = F.mse_loss(
            model(inputs), coll.construct_cloud_targets(indices1)
        )
        # loss = _pde_loss + bc_loss
        # loss.backward()

    total_pde_loss_u = ff_u[0]
    for i in range(1, len(ff_u)):
        total_pde_loss_u += ff_u[i]

    total_pde_loss_u = total_pde_loss_u / len(ff_u)

    total_pde_loss_v = ff_v[0]
    for i in range(1, len(ff_v)):
        total_pde_loss_v += ff_v[i]

    total_pde_loss_v = total_pde_loss_v / len(ff_v)

    total_pde_loss_mass = ff_continuty[0]
    for i in range(1, len(ff_continuty)):
        total_pde_loss_mass += ff_continuty[i]

    total_pde_loss_mass = total_pde_loss_mass / len(ff_continuty)

    total_bc = bbc[0]
    for i in range(1, len(bbc)):
        total_bc += bbc[i]

    total_bc = total_bc / len(bbc)
    bc_loss = total_bc

    _pde_loss = total_pde_loss_u + total_pde_loss_v + total_pde_loss_mass

    loss = _pde_loss + bc_loss
    loss.backward()

    optimizer.step()
    if config.use_lrscheduler:
        lr_scheduler.step()
    return {
        "total_loss": loss,
        "bc_loss": bc_loss,
        "pde_loss": _pde_loss,
        "collocation_loss": collocation_loss,
    }


def train_batch_data_driven(
    coll_points_raw,
    boundary_points_raw,
    inlets_raw,
    outlets_raw,
    obstacles_raw,
    walls_raw,
    model,
    optimizer,
    lr_scheduler,
    criterion,
    ni,
    extra_variables,
    config,
):

    model.train()
    optimizer.zero_grad()

    coll = {"x": [], "y": [], "u": [], "v": [], "p": []}
    bc = {"x": [], "y": [], "u": [], "v": [], "p": []}

    # infox = {"obstacle": 262, "wall": 404, "inlet": 202, "outlet": 202}

    for coll_points, boundary_points, inlets, outlets, obstacles, walls in zip(
        coll_points_raw,
        boundary_points_raw,
        inlets_raw,
        outlets_raw,
        obstacles_raw,
        walls_raw,
    ):
        _inputs, x, y = coll_points.construct_inputs()
        _outputs = coll_points.construct_targets()

        (
            bc_inputs,
            _,
            _,
        ) = boundary_points.construct_inputs()
        bc_outputs = boundary_points.construct_targets()

        preds = model(_inputs)
        loss = F.mse_loss(preds, _outputs) + F.mse_loss(model(bc_inputs), bc_outputs)
        bc_loss = _pde_loss = collocation_loss = 0
        loss.backward()

    optimizer.step()
    if config.use_lrscheduler:
        lr_scheduler.step()
    return {
        "total_loss": loss,
        "bc_loss": bc_loss,
        "pde_loss": _pde_loss,
        "collocation_loss": collocation_loss,
    }


def train(
    g,
    model,
    train_loader: GeometryDataLoader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    lr_scheduler,
    config,
    ni,
    extra_variables,
):

    wandb.watch(model, criterion, log="all", log_freq=10)

    example_ct = 0
    batch_ct = 0
    best_loss = 10e9

    val_loss = None
    pbar = tqdm(range(config.epochs))
    training_logs = []
    RESULT_SAVE_DIR = os.path.join(config.mplsave_dir, "results")
    CHECKPOINT_SAVE_DIR = os.path.join(config.mplsave_dir, "checkpoints")
    RESULT_SAVE_PATH = os.path.join(
        RESULT_SAVE_DIR, f"{os.path.join(os.path.basename(config.dataset_dir))}.json"
    )
    CHECKPOINT_SAVE_PATH = os.path.join(
        CHECKPOINT_SAVE_DIR, f"{os.path.join(os.path.basename(config.dataset_dir))}.pth"
    )
    Path(RESULT_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINT_SAVE_DIR).mkdir(parents=True, exist_ok=True)

    if config.training_scheme == "data":
        train_fn = train_batch_data_driven
    elif config.training_scheme == "pinn":
        train_fn = (
            train_batch_pinn_single
            if train_loader.dataset_size == 1
            else train_batch_pinn_multiplev2
        )

    for epoch in pbar:
        for cfddata in train_loader:
            info = train_fn(
                cfddata,
                model,
                optimizer,
                lr_scheduler,
                criterion,
                ni,
                extra_variables,
                config,
                g,
                epoch,
            )
            example_ct += len(cfddata.dd_collocation)
            batch_ct += 1
            if ((batch_ct + 1) % 1) == 0:
                metrics = ["total_loss", "pde_loss", "collocation_loss", "bc_loss"]
                for m in metrics:
                    wandb.log({"epoch": epoch, m: info[m]}, step=example_ct)

        # validate
        if epoch % config.validation_step == 0:
            loss_dict, val_outputs = validate(
                model, val_loader, criterion, epoch, config, ni
            )
            wandb.log({"val_loss": loss_dict["avg.mse"], "epoch": epoch})

            # predict(model, test_loader, criterion, epoch, ni)
            if loss_dict["avg.mse"] < best_loss:
                best_loss = loss_dict["avg.mse"]
                with open(RESULT_SAVE_PATH, "w") as fp:
                    json.dump(loss_dict, fp, indent=4)

        losses_info = {
            m: info[m].item()
            for m in ["total_loss", "pde_loss", "collocation_loss", "bc_loss"]
        }
        training_logs.append(
            {
                "epoch": epoch,
                "u": loss_dict["avg.u"],
                "v": loss_dict["avg.v"],
                "p": loss_dict["avg.p"],
                "val_loss": loss_dict["avg.mse"],
            }
            | losses_info
        )

        # save weight
        if epoch % config.save_pth_epoch == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": info["total_loss"],
                },
                CHECKPOINT_SAVE_PATH,
            )

        val_loss = loss_dict["avg.mse"]
        pbar.set_description(
            f"**train**| Loss/ total: {info['total_loss']:.3e}, pde: {info['pde_loss']:.3e}, bc: {info['bc_loss']:.3e}, coll: {info['collocation_loss']:.3e}  ~ **val** | loss: {val_loss:.3e}"
        )

    # with open(os.path.join(config.mplsave_dir, f"training_logs.csv"), "w") as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=training_logs[0].keys())
    #     writer.writeheader()
    #     writer.writerows(training_logs)


def validate(model, val_loader, criterion, epoch, config, ni):
    model.eval()
    losses = []
    l2_errors = []

    loss_dict = {"l2_u_error": [], "l2_v_error": [], "l2_p_error": [], "mse_error": []}
    for i, cfddata in enumerate(val_loader):
        coll_points = cfddata.dd_collocation[0]  # (only working with the first one)
        boundary_points = cfddata.dd_boundary[0]
        with torch.no_grad():
            inputs, x, y = coll_points.construct_inputs()
            targets = coll_points.construct_targets()

            preds = model(inputs)
            loss = criterion(preds, targets)

            u_error = (
                norm(targets[:, 0, :] - preds[:, 0, :], ord=2)
                / norm(targets[:, 0, :], ord=2)
            ).item() * 100

            v_error = (
                norm(targets[:, 1, :] - preds[:, 1, :], ord=2)
                / norm(targets[:, 1, :], ord=2)
            ).item() * 100

            p_error = (
                norm(targets[:, 2, :] - preds[:, 2, :], ord=2)
                / norm(targets[:, 2, :], ord=2)
            ).item() * 100

            loss_dict["l2_u_error"].append(u_error)
            loss_dict["l2_v_error"].append(v_error)
            loss_dict["l2_p_error"].append(p_error)
            loss_dict["mse_error"].append(loss.item())

        break
    assert len(loss_dict["mse_error"]) == 1
    loss_dict["avg.u"] = sum(loss_dict["l2_u_error"]) / len(loss_dict["l2_u_error"])
    loss_dict["avg.v"] = sum(loss_dict["l2_v_error"]) / len(loss_dict["l2_v_error"])
    loss_dict["avg.p"] = sum(loss_dict["l2_p_error"]) / len(loss_dict["l2_p_error"])
    loss_dict["avg.mse"] = sum(loss_dict["mse_error"]) / len(loss_dict["mse_error"])
    info = {"x": x, "y": y, "exact": coll_points, "preds": preds}
    return loss_dict, info


def test(model, test_val, criterion, config):
    model.eval()
    losses = []
    l2_errors = []

    loss_dict = {"l2_u_error": [], "l2_v_error": [], "l2_p_error": [], "mse_error": []}
    for i, cfddata in enumerate(test_val):
        coll_points = cfddata.dd_collocation[0]  # (only working with the first one)
        boundary_points = cfddata.dd_boundary[0]
        with torch.no_grad():
            inputs, x, y = coll_points.construct_inputs()
            targets = coll_points.construct_targets()

            preds = model(inputs)
            loss = criterion(preds, targets)
            u_error = (
                norm(targets[:, 0, :] - preds[:, 0, :], ord=2)
                / norm(targets[:, 0, :], ord=2)
            ).item() * 100

            v_error = (
                norm(targets[:, 1, :] - preds[:, 1, :], ord=2)
                / norm(targets[:, 1, :], ord=2)
            ).item() * 100

            p_error = (
                norm(targets[:, 2, :] - preds[:, 2, :], ord=2)
                / norm(targets[:, 2, :], ord=2)
            ).item() * 100

            loss_dict["l2_u_error"].append(u_error)
            loss_dict["l2_v_error"].append(v_error)
            loss_dict["l2_p_error"].append(p_error)
            loss_dict["mse_error"].append(loss.item())
            # import pudb; pu.db
        break
    assert len(loss_dict["mse_error"]) == 1
    loss_dict["avg.u"] = sum(loss_dict["l2_u_error"]) / len(loss_dict["l2_u_error"])
    loss_dict["avg.v"] = sum(loss_dict["l2_v_error"]) / len(loss_dict["l2_v_error"])
    loss_dict["avg.p"] = sum(loss_dict["l2_p_error"]) / len(loss_dict["l2_p_error"])
    loss_dict["avg.mse"] = sum(loss_dict["mse_error"]) / len(loss_dict["mse_error"])

    info = {"x": x, "y": y, "exact": coll_points, "preds": preds}
    x_coord = info["x"].squeeze()
    y_coord = info["y"].squeeze()
    exact_u = info["exact"].u[:, :, :].squeeze()
    exact_v = info["exact"].v[:, :, :].squeeze()
    exact_p = info["exact"].p[:, :, :].squeeze()

    predicted_u = info["preds"][:, 0].squeeze()
    predicted_v = info["preds"][:, 1].squeeze()
    predicted_p = info["preds"][:, 2].squeeze()

    plot_solution(
        x_coord=x_coord,
        y_coord=y_coord,
        exact_u=exact_u,
        exact_v=exact_v,
        exact_p=exact_p,
        predicted_u=predicted_u,
        predicted_v=predicted_v,
        predicted_p=predicted_p,
        fig_title=f"MSE: {loss_dict['mse_error_avg']:.3e}",
        save_path=os.path.join(config.mplsave_dir, f"best.png"),
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def model_pipeline(cfg: DictConfig):
    # print(cfg.pretty())
    print(OmegaConf.to_yaml(cfg))
    with wandb.init(
        project="dd-cfd",
        config=OmegaConf.to_container(cfg.scheme, resolve=True, throw_on_missing=True),
    ):
        config = wandb.config
        Path(config.mplsave_dir).mkdir(parents=True, exist_ok=True)

        if config.load_data_mode == "normal":
            train_dir = os.path.join(config.dataset_dir, "train/*")
            val_dir = os.path.join(config.dataset_dir, "test/*")
            cfd_domains_train = create_cfddomain_set(train_dir, device)
            cfd_domains_val = create_cfddomain_set(val_dir, device)
            train_loader = GeometryDataLoader(
                cfd_domains_train, config.batch_size, True
            )
            val_loader = GeometryDataLoader(cfd_domains_val, config.batch_size, False)
            test_loader = val_loader

        elif config.load_data_mode == "share":
            train_dir = os.path.join(config.dataset_dir)
            cfd_domains_train = create_cfddomain_set(train_dir, device)
            train_loader = GeometryDataLoader(cfd_domains_train, 1, True)
            val_loader = GeometryDataLoader(cfd_domains_train, 1, False)
            test_loader = val_loader

        criterion = torch.nn.MSELoss()
        extra_variables = dict()
        extra_variables["l1"] = nn.Parameter(
            torch.tensor(1.0, device=device), requires_grad=True
        )
        extra_variables["l2"] = nn.Parameter(
            torch.tensor(0.01, device=device), requires_grad=True
        )
        print(f"use {device} for training")

        # load teacher network (if available)
        if config.g_enable:
            from models.base import PointNetSeg

            g_teacher = PointNetSeg()
            g_teacher = g_teacher.to(device)
            g_pretrained = torch.load(config.g_pretrained)
            g_teacher.load_state_dict(g_pretrained["model_state_dict"])
            _epoch = g_pretrained["epoch"]
            _loss = g_pretrained["loss"]
            print("g_teacher: \n", g_teacher)
            print(f"g_teacher: load from {_epoch} with loss: {_loss} sucessfully !")
        else:
            g_teacher = None

        # instantiate student network
        f_model = hydra.utils.instantiate(config.network)
        f_model = f_model.to(device)
        print(f_model)
        print(f"num of training parameters: {count_parameters(f_model)}")

        if config.f_pretrained != None:
            f_pretrained = torch.load(config.f_pretrained)
            f_model.load_state_dict(f_pretrained["model_state_dict"])
            optimizer.load_state_dict(f_pretrained["optimizer_state_dict"])
            _epoch = f_pretrained["epoch"]
            _loss = f_pretrained["loss"]
            print(f"load checkpoint from {_epoch} with loss: {_loss} sucessfully !")

        # setup optimizer
        optimizer = hydra.utils.instantiate(config.optimizer)(f_model.parameters())

        # setup learning rate scheduler
        if config.lr_scheduler_enable:
            lr_scheduler = hydra.utils.instantiate(config.lr_scheduler)(optimizer)
        else:
            lr_scheduler = None

        if config.mode == "train":
            train(
                g_teacher,
                f_model,
                train_loader,
                val_loader,
                test_loader,
                criterion,
                optimizer,
                lr_scheduler,
                config,
                [],
                extra_variables,
            )
            # validate(model, test_loader, criterion, epoch=0, config=config, ni=None)

        if config.mode == "test":
            # test(f_model, test_loader, criterion, config)
            pass


if __name__ == "__main__":
    model_pipeline()
