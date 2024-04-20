from functools import partial
import h5py
import torch
from torch.utils.data import DataLoader
import pinnstorch.data
import pinnstorch.data
import pointnet
import wandb
import hydra
import numpy as np
from tqdm import tqdm
from pathlib import Path


import matplotlib.pyplot as plt
import matplotlib
from torch.nn import functional as F
from pinnstorch.utils.gradient_fn import gradient
from pinnstorch.data import GeometryDataLoader
import hydra
from omegaconf import DictConfig, OmegaConf


matplotlib.use("Agg")

# config = dict(
#     learing_rate=0.001,
#     batch_size=4,
#     epochs=5000,
#     optimizer=dict(_target_=torch.optim.Adam, lr=0.001, eps=1e-6, _partial_=True),
#     m=3,
#     scheduler=None,
#     validation_step=1,
#     num_cells=5000,
#     savefig_path="plots-server3-datadriven2/",
#     dataset_path="/home/pham/code/ds-07.hdf5",
#     save_pth_epoch=10,
#     save_pth_path="pretrained/mf.pth",
# )

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_dataset(path=None, split=[0.8, 0.2]):
    hdf5_file = h5py.File(path, "r")
    geo_data = hdf5_file["simdata"][:, ...]
    metadata = hdf5_file["metadata"][:, ...]

    u_min = np.min(geo_data[:, :, 2])
    u_max = np.max(geo_data[:, :, 2])
    v_min = np.min(geo_data[:, :, 3])
    v_max = np.max(geo_data[:, :, 3])
    p_min = np.min(geo_data[:, :, 4])
    p_max = np.max(geo_data[:, :, 4])

    ni = {
        "u_max": u_max,
        "u_min": u_min,
        "v_max": v_max,
        "v_min": v_min,
        "p_max": p_max,
        "p_min": p_min,
    }

    assert geo_data.shape[-1] == 5
    geo_data[:, :, 2] = (geo_data[:, :, 2] - u_min) / (u_max - u_min)
    geo_data[:, :, 3] = (geo_data[:, :, 3] - v_min) / (v_max - v_min)
    geo_data[:, :, 4] = (geo_data[:, :, 4] - p_min) / (p_max - p_min)

    train_len = int(split[0] * geo_data.shape[0])
    test_len = int(split[0] * geo_data.shape[0])

    train_set = geo_data[:train_len, :, :]
    test_set = geo_data[train_len : (train_len + test_len), :, :]

    metadata_train = metadata[:train_len, :, :]
    metadata_test = metadata[train_len : (train_len + test_len), :, :]
    train_set = torch.permute(torch.tensor(train_set), (0, 2, 1))
    test_set = torch.permute(torch.tensor(test_set), (0, 2, 1))

    return train_set, test_set, metadata_train, metadata_test, ni


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make(config):

    model = pointnet.PointNetSegHead(num_points=config.num_cells, m=config.m)
    model = model.to(device)
    print(f"num of training parameters: {count_parameters(model)}")
    print(f"use {device} for training")

    train_set, test_set, metadata_train, metadata_test, normalization_info = (
        read_dataset(config.dataset_path)
    )

    train_loader = GeometryDataLoader(
        geometries=train_set,
        metadata=metadata_train,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = GeometryDataLoader(
        geometries=test_set,
        metadata=metadata_test,
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = GeometryDataLoader(
        geometries=test_set[:1, :, :],
        metadata=metadata_test[:1, :, :],
        batch_size=1,
        shuffle=False,
    )

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

    if config.pretrained_model != None:
        checkpoint = torch.load(config.pretrained_model)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(f"load checkpoint from {epoch} with loss: {loss} sucessfully !")

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


def train_batch(
    X, targets, model, optimizer, criterion, ni, extra_variables, config, metadata
):
    X, targets = X.to(device), targets.to(device)
    metadata = torch.tensor(metadata, dtype=torch.bool).to(device)

    model.train()
    optimizer.zero_grad()

    x, y = torch.split(X, split_size_or_sections=1, dim=1)
    x.requires_grad = True
    x.retain_grad()

    y.requires_grad = True
    y.retain_grad()

    inputs = torch.cat([x, y], dim=1)
    preds, _, _ = model(inputs)
    interior_preds = preds[:, :, :]
    loss = pde_loss(x, y, interior_preds, extra_variables) + F.mse_loss(
        torch.masked_select(targets, ~metadata), torch.masked_select(preds, ~metadata)
    )
    # loss = F.mse_loss(preds, targets)
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
    # total_batches = len(train_loader) * config.epochs

    example_ct = 0  # number of examples seen
    batch_ct = 0

    for epoch in tqdm(range(config.epochs)):
        print("Epoch: ", epoch)
        for train_data, metadata in train_loader:
            x_train = train_data[:, 0:2, :]

            targets = train_data[:, 2:, :]
            targets = torch.permute(targets, (0, 2, 1))

            loss = train_batch(
                x_train,
                targets,
                model,
                optimizer,
                criterion,
                ni,
                extra_variables,
                config,
                metadata,
            )
            example_ct += len(x_train)
            batch_ct += 1

            if ((batch_ct + 1) % 1) == 0:
                train_log(loss, example_ct, epoch)

        # validate
        if epoch % config.validation_step == 0:
            validate(model, val_loader, criterion, epoch, config, ni)
            # predict(model, test_loader, criterion, epoch, ni)

        # save weight
        if epoch % config.save_pth_epoch == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                config.save_pth_path,
            )


def train_log(loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "train_loss": loss}, step=example_ct)
    print(f"train_loss after {str(example_ct).zfill(6)} examples: {loss:.4f}")


def validate(model, val_loader, criterion, epoch, config, ni):
    model.eval()
    losses = []

    for i, (val_data, metadata) in enumerate(val_loader):
        x_val = val_data[:, 0:2, :]
        targets = val_data[:, 2:, :]
        targets = torch.permute(targets, (0, 2, 1))
        x_val, targets = x_val.to(device), targets.to(device)

        with torch.no_grad():
            x_val.requires_grad = True
            x_val.retain_grad()
            pred, _, _ = model(x_val)
            loss = criterion(pred, targets)
            losses.append(loss.item())

    wandb.log({"val_loss": sum(losses) / len(losses), "epoch": epoch})
    print(f"val_loss: {sum(losses)/len(losses):.4f}, epoch: {epoch}")

    iid = 0
    save_path = config.savefig_path + str(epoch) + ".png"
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def model_pipeline(cfg: DictConfig):
    # run = wandb.init(entity=cfg, project="dd-cfd")
    print(cfg)
    with wandb.init(
        project="dd-cfd",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    ):
        config = wandb.config

        Path(cfg.savefig_path).mkdir(parents=True, exist_ok=True)
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
        # print(model)

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

    # with wandb.init(project="dd-cfd", config=cfg):
    #     config = wandb.config


if __name__ == "__main__":
    model_pipeline()
