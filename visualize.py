from typing import Any, Union

import torch
import matplotlib.pyplot as plt
import matplotlib

import numpy as np

matplotlib.use("Agg")


def visualize(
    x_coord: Union[np.array, torch.tensor] = [],
    y_coord: Union[np.array, torch.tensor] = [],
    exact_u: Union[np.array, torch.tensor] = [],
    exact_v: Union[np.array, torch.tensor] = [],
    exact_p: Union[np.array, torch.tensor] = [],
    predicted_u: Union[np.array, torch.tensor] = [],
    predicted_v: Union[np.array, torch.tensor] = [],
    predicted_p: Union[np.array, torch.tensor] = [],
    fig_title: str = "__cmp__",
    save_path: str = "fig.png",
    kwargs: dict = {},
):
    assert x_coord.shape[-1] > 500
    # extent = -0.25, 0.65, -0.1, 0.1

    if torch.is_tensor(x_coord):
        x_coord = x_coord.cpu().detach().numpy()
    if torch.is_tensor(y_coord):
        y_coord = y_coord.cpu().detach().numpy()
    if torch.is_tensor(exact_u):
        exact_u = exact_u.cpu().detach().numpy()
    if torch.is_tensor(exact_v):
        exact_v = exact_v.cpu().detach().numpy()
    if torch.is_tensor(exact_p):
        exact_p = exact_p.cpu().detach().numpy()

    if torch.is_tensor(predicted_u):
        predicted_u = predicted_u.cpu().detach().numpy()
    if torch.is_tensor(predicted_v):
        predicted_v = predicted_v.cpu().detach().numpy()
    if torch.is_tensor(predicted_p):
        predicted_p = predicted_p.cpu().detach().numpy()

    plt.figure(figsize=(5, 12))
    plt.suptitle(fig_title)

    plt.subplot(311)
    plt.ylabel("p:exact $(Pa)$", fontsize=15)
    plt.scatter(x_coord, y_coord, c=exact_p, cmap="jet")

    plt.subplot(312)
    plt.ylabel("p:predicted $(Pa)$", fontsize=15)
    plt.scatter(x_coord, y_coord, c=predicted_p, cmap="jet")

    plt.subplot(313)
    plt.ylabel("p:difference", fontsize=15)
    plt.scatter(x_coord, y_coord, c=np.abs(predicted_p - exact_p), cmap="jet")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_solution(
    x_coord: Union[np.array, torch.tensor] = [],
    y_coord: Union[np.array, torch.tensor] = [],
    exact_u: Union[np.array, torch.tensor] = [],
    exact_v: Union[np.array, torch.tensor] = [],
    exact_p: Union[np.array, torch.tensor] = [],
    predicted_u: Union[np.array, torch.tensor] = [],
    predicted_v: Union[np.array, torch.tensor] = [],
    predicted_p: Union[np.array, torch.tensor] = [],
    fig_title: str = "__cmp__",
    save_path: str = "fig.png",
    kwargs: dict = {},
):
    assert x_coord.shape[-1] > 500
    # extent = -0.25, 0.65, -0.1, 0.1

    if torch.is_tensor(x_coord):
        x_coord = x_coord.cpu().detach().numpy()
    if torch.is_tensor(y_coord):
        y_coord = y_coord.cpu().detach().numpy()
    if torch.is_tensor(exact_u):
        exact_u = exact_u.cpu().detach().numpy()
    if torch.is_tensor(exact_v):
        exact_v = exact_v.cpu().detach().numpy()
    if torch.is_tensor(exact_p):
        exact_p = exact_p.cpu().detach().numpy()

    if torch.is_tensor(predicted_u):
        predicted_u = predicted_u.cpu().detach().numpy()
    if torch.is_tensor(predicted_v):
        predicted_v = predicted_v.cpu().detach().numpy()
    if torch.is_tensor(predicted_p):
        predicted_p = predicted_p.cpu().detach().numpy()

    plt.figure(figsize=(15, 12))
    # plt.suptitle(fig_title)
    plt.tight_layout()

    # u
    plt.subplot(331)
    plt.margins(0, 0)
    plt.gca().set_title("u:exact $(m/s)$", fontsize=25)
    exact_u_scatter = plt.scatter(x_coord, y_coord, c=exact_u, cmap="jet")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    cbar = plt.colorbar(exact_u_scatter)
    cbar.ax.tick_params(labelsize=15)

    plt.margins(0, 0)

    plt.subplot(332)
    plt.margins(0, 0)
    plt.gca().set_title("u:predicted $(m/s)$", fontsize=25)
    predicted_u_scatter = plt.scatter(x_coord, y_coord, c=predicted_u, cmap="jet")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    cbar = plt.colorbar(predicted_u_scatter)
    cbar.ax.tick_params(labelsize=15)

    plt.subplot(333)
    plt.margins(0, 0)
    plt.gca().set_title("u:difference", fontsize=25)
    diff_scatter = plt.scatter(
        x_coord, y_coord, c=np.abs(predicted_u - exact_u), cmap="jet"
    )
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    cbar = plt.colorbar(diff_scatter)
    cbar.ax.tick_params(labelsize=15)

    # v
    plt.subplot(334)
    plt.margins(0, 0)
    plt.gca().set_title("v:exact $(m/s)$", fontsize=25)
    exact_v_scatter = plt.scatter(x_coord, y_coord, c=exact_v, cmap="jet")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    cbar = plt.colorbar(exact_v_scatter)
    cbar.ax.tick_params(labelsize=15)

    plt.subplot(335)
    plt.margins(0, 0)
    plt.gca().set_title("v:predicted $(m/s)$", fontsize=25)
    predicted_v_scatter = plt.scatter(x_coord, y_coord, c=predicted_v, cmap="jet")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    cbar = plt.colorbar(predicted_v_scatter)
    cbar.ax.tick_params(labelsize=15)

    plt.subplot(336)
    plt.margins(0, 0)
    plt.gca().set_title("v:difference", fontsize=25)
    diff_scatter = plt.scatter(
        x_coord, y_coord, c=np.abs(predicted_v - exact_v), cmap="jet"
    )
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    cbar = plt.colorbar(diff_scatter)
    cbar.ax.tick_params(labelsize=15)

    plt.subplot(337)
    plt.margins(0, 0)
    plt.gca().set_title("p:exact $(Pa)$", fontsize=25)
    exact_p_scatter = plt.scatter(x_coord, y_coord, c=exact_p, cmap="jet")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    cbar = plt.colorbar(exact_p_scatter)
    cbar.ax.tick_params(labelsize=15)

    plt.subplot(338)
    plt.margins(0, 0)
    plt.gca().set_title("p:predicted $(Pa)$", fontsize=25)
    predicted_p_scatter = plt.scatter(x_coord, y_coord, c=predicted_p, cmap="jet")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    cbar = plt.colorbar(predicted_p_scatter)
    cbar.ax.tick_params(labelsize=15)

    plt.subplot(339)
    plt.margins(0, 0)
    plt.gca().set_title("p:difference", fontsize=25)
    diff_scatter = plt.scatter(
        x_coord, y_coord, c=np.abs(predicted_p - exact_p), cmap="jet"
    )
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    cbar = plt.colorbar(diff_scatter)
    cbar.ax.tick_params(labelsize=15)

    plt.margins(0, 0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".png", ".pdf"))

    plt.close()


def visualize_single(
    x_coord: Union[np.array, torch.tensor] = [],
    y_coord: Union[np.array, torch.tensor] = [],
    exact_u: Union[np.array, torch.tensor] = [],
    exact_v: Union[np.array, torch.tensor] = [],
    exact_p: Union[np.array, torch.tensor] = [],
    fig_title: str = "__cmp__",
    save_path: str = "fig.png",
    kwargs: dict = {},
):
    assert x_coord.shape[-1] > 500
    # extent = -0.25, 0.65, -0.1, 0.1

    if torch.is_tensor(x_coord):
        x_coord = x_coord.cpu().detach().numpy()
    if torch.is_tensor(y_coord):
        y_coord = y_coord.cpu().detach().numpy()
    if torch.is_tensor(exact_u):
        exact_u = exact_u.cpu().detach().numpy()
    if torch.is_tensor(exact_v):
        exact_v = exact_v.cpu().detach().numpy()
    if torch.is_tensor(exact_p):
        exact_p = exact_p.cpu().detach().numpy()

    plt.figure(figsize=(10, 4))
    plt.suptitle(fig_title)

    plt.subplot(111)
    plt.ylabel("p:exact", fontsize=15)
    plt.scatter(x_coord, y_coord, c=exact_p, cmap="jet")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
