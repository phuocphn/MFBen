from functools import partial
import h5py
import pdb
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import pointnet
import wandb
import hydra
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from torch.nn import functional as F
from pinnstorch.utils.gradient_fn import gradient


matplotlib.use("Agg")
from omegaconf import DictConfig, OmegaConf

config = dict(
    learing_rate=0.001,  # 5e-4,
    batch_size=16,
    epochs=5000,
    # optimizer=dict(_target_=torch.optim.LBFGS, lr=0.001, max_iter=5000, _partial_=True),
    optimizer=dict(_target_=torch.optim.Adam, lr=0.001, eps=1e-6, _partial_=True),
    # optimizer=dict(_target_=torch.optim.SGD, lr=0.001, _partial_=True),
    m=3,
    scheduler=None,
    validation_step=1,
    num_cells=int(2000),
    dataset_path="/mnt/home/pham/data-gen/MLCAD/ds-04/2dobs+fixFluidType+fixShape.hdf5",
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_dataset(path=None, split=[0.8, 0.2]):
    hdf5_file = h5py.File(path, "r")
    data = hdf5_file["interior_data"][:, ...]

    ni = {
        "u_max": 0.0528556,
        "u_min": -0.0104957,
        "v_max": 0.0263741,
        "v_min": -0.026564,
        "p_max": 0.0014619,
        "p_min": -0.000719237,
    }
    u_min, u_max = ni["u_min"], ni["u_max"]
    v_min, v_max = ni["v_min"], ni["v_max"]
    p_min, p_max = ni["p_min"], ni["p_max"]

    assert data.shape[-1] == 5
    data[:, :, 2] = (data[:, :, 2] - u_min) / (u_max - u_min)
    data[:, :, 3] = (data[:, :, 3] - v_min) / (v_max - v_min)
    data[:, :, 4] = (data[:, :, 4] - p_min) / (p_max - p_min)

    train_len = int(split[0] * data.shape[0])
    test_len = int(split[0] * data.shape[0])

    train_set = data[:train_len, :, :]
    test_set = data[train_len : (train_len + test_len), :, :]
    train_set = torch.permute(torch.tensor(train_set), (0, 2, 1))
    test_set = torch.permute(torch.tensor(test_set), (0, 2, 1))

    return train_set, test_set, ni


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make(config):

    model = pointnet.PointNetSegHead(num_points=config.num_cells, m=config.m)
    # from net import Tnet
    # model = Tnet()
    model = model.to(device)
    print(f"num of training parameters: {count_parameters(model)}")
    print(f"use {device} for training")

    train_set, test_set, normalization_info = read_dataset(config.dataset_path)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_set[:1, :, :], batch_size=1, shuffle=False)

    criterion = torch.nn.MSELoss()
    extra_variables = dict()
    extra_variables["l1"] = torch.nn.Parameter(
        torch.tensor(1.0, device=device),
        requires_grad=True,
    )
    extra_variables["l2"] = torch.nn.Parameter(
        torch.tensor(0.01, device=device),
        requires_grad=True,
    )

    optimizer = hydra.utils.instantiate(config.optimizer)(
        params=list(model.parameters()) + [extra_variables["l1"], extra_variables["l2"]]
    )

    # optimizer = optim.Adam(
    #     list(model.parameters()) + [extra_variables["l1"], extra_variables["l2"]],
    #     lr=config.learing_rate,
    #     eps=1e-6,
    # )
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3,
    #                                          step_size_up=1000, cycle_momentum=False)

    return (
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        normalization_info,
        extra_variables,
    )


# from torch import nn, autograd, Tensor


def pde_loss(x, y, preds, extra_variables):
    assert preds.shape[-1] == 3
    assert x.shape[-2] == 1

    u_pred = preds[:, :, 0:1]
    v_pred = preds[:, :, 1:2]
    p_pred = preds[:, :, 2:3]

    u_x, u_y = gradient(u_pred, [x, y])
    u_xx = gradient(u_x, x)[0]
    u_yy = gradient(u_y, y)[0]

    v_x, v_y = gradient(v_pred, [x, y])
    v_xx = gradient(v_x, x)[0]
    v_yy = gradient(v_y, y)[0]

    p_x, p_y = gradient(p_pred, [x, y])

    r_continuity = u_x + v_x

    f_u = (
        extra_variables["l1"] * (u_pred * u_x + v_pred * u_y)
        + p_x
        - extra_variables["l2"] * (u_xx + u_yy)
    )
    f_v = (
        extra_variables["l1"] * (u_pred * v_x + v_pred * v_y)
        + p_y
        - extra_variables["l2"] * (v_xx + v_yy)
    )

    f_u_loss = F.mse_loss(f_u, torch.zeros_like(f_u))
    f_v_loss = F.mse_loss(f_v, torch.zeros_like(f_v))
    f_continuity = F.mse_loss(r_continuity, torch.zeros_like(r_continuity))
    loss = f_u_loss + f_v_loss + f_continuity
    return loss


def train_batch(X, targets, model, optimizer, criterion, ni, extra_variables):
    X, targets = X.to(device), targets.to(device)

    model.train()
    optimizer.zero_grad()

    x, y = torch.split(X, split_size_or_sections=1, dim=1)
    x.requires_grad = True
    x.retain_grad()

    y.requires_grad = True
    y.retain_grad()

    inputs = torch.cat([x, y], dim=1)
    pred, _, _ = model(inputs)
    loss = pde_loss(x, y, pred, extra_variables)

    loss.backward()
    optimizer.step()

    return loss


def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    config,
    ni,
    extra_variables,
):

    wandb.watch(model, criterion, log="all", log_freq=10)
    total_batches = len(train_loader) * config.epochs

    example_ct = 0  # number of examples seen
    batch_ct = 0

    for epoch in tqdm(range(config.epochs)):
        print("Epoch: ", epoch)
        for train_data in train_loader:
            x_train = train_data[:, 0:2, : config.num_cells]

            targets = train_data[:, 2:, : config.num_cells]
            targets = torch.permute(targets, (0, 2, 1))

            loss = train_batch(
                x_train, targets, model, optimizer, criterion, ni, extra_variables
            )
            example_ct += len(x_train)
            batch_ct += 1

            if ((batch_ct + 1) % 1) == 0:
                train_log(loss, example_ct, epoch)

        # validate
        if epoch % config.validation_step == 0:
            validate(model, val_loader, criterion, epoch, config, ni)
            # predict(model, test_loader, criterion, epoch, ni)


def train_log(loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "train_loss": loss}, step=example_ct)
    print(f"train_loss after {str(example_ct).zfill(6)} examples: {loss:.4f}")


def validate(model, val_loader, criterion, epoch, config, ni):
    model.eval()
    losses = utils.AverageMeter("Validation Loss:", ":.4e")

    for i, val_data in enumerate(val_loader):
        x_val = val_data[:, 0:2, : config.num_cells]
        targets = val_data[:, 2:, : config.num_cells]
        targets = torch.permute(targets, (0, 2, 1))
        x_val, targets = x_val.to(device), targets.to(device)

        with torch.no_grad():
            x_val.requires_grad = True
            x_val.retain_grad()
            pred, _, _ = model(x_val)
            loss = criterion(pred, targets)
            losses.update(loss.item(), x_val.size(0))

    wandb.log({"val_loss": losses.avg, "epoch": epoch})
    print(f"val_loss: {losses.avg:.4f}, epoch: {epoch}")

    iid = 0
    save_path = "plots/" + str(epoch) + ".png"
    dump_prediction(
        x_coord=x_val.transpose(2, 1)[iid, :, -2].cpu(),
        y_coord=x_val.transpose(2, 1)[iid, :, -1].cpu(),
        y_true=targets[iid, :, :].cpu(),
        y_pred=pred[iid, :, :].cpu(),
        loss=loss,
        config=config,
        save_path=save_path,
    )


def dump_prediction(x_coord, y_coord, y_true, y_pred, loss, config, save_path):
    extent = -0.25, 0.65, -0.1, 0.1
    plt.suptitle(
        "Comparision of OpenFOAM vs Deep Learning\nMean Squared Error: {0:0.5f}".format(
            loss.item()
        ),
        fontsize=13,
    )
    plt.subplot(211)

    ux_true = y_true[: config.num_cells, 0].detach().numpy()
    uy_true = y_true[: config.num_cells, 1].detach().numpy()
    p_true = y_true[: config.num_cells, 2].detach().numpy()

    ux_pred = y_pred[: config.num_cells, 0].detach().numpy()
    uy_pred = y_pred[: config.num_cells, 1].detach().numpy()
    p_pred = y_pred[: config.num_cells, 2].detach().numpy()

    x_coord = x_coord[: config.num_cells].detach().numpy()
    y_coord = y_coord[: config.num_cells].detach().numpy()

    plt.ylabel("OpenFOAM", fontsize=15)
    plt.scatter(x_coord, y_coord, c=p_true, cmap="viridis")

    plt.subplot(212)
    plt.ylabel("PINN", fontsize=15)
    plt.scatter(x_coord, y_coord, c=p_pred, cmap="viridis")

    plt.subplots_adjust(left=0.2, wspace=0.8, top=0.85)
    plt.savefig(save_path)
    plt.close()


def model_pipeline(hyperparameters):
    with wandb.init(project="dd-cfd", config=hyperparameters):
        config = wandb.config

        # make the model, data, and optimization problem
        (
            model,
            train_loader,
            val_loader,
            test_loader,
            criterion,
            optimizer,
            normalization_info,
            extra_variables,
        ) = make(config)
        print(model)

        train(
            model,
            train_loader,
            val_loader,
            test_loader,
            criterion,
            optimizer,
            config,
            normalization_info,
            extra_variables,
        )

    return model


model_pipeline(config)
