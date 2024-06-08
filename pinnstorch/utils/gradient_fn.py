from typing import Dict, List, Union, Optional, Tuple

import torch


@torch.jit.script
def gradient(
    dy: torch.Tensor,
    dx: Union[List[torch.Tensor], torch.Tensor],
    ones_like_tensor: Optional[List[Optional[torch.Tensor]]] = None,
    create_graph: bool = True,
) -> List[torch.Tensor]:
    """Compute the gradient of a tensor `dy` with respect to another tensor `dx`.

    :param dy: The tensor to compute the gradient for.
    :param dx: The tensor with respect to which the gradient is computed.
    :param ones_like_tensor: A tensor with the same shape as `dy`, used for creating the gradient (default is None).
    :param create_graph: Whether to create a computationa graph for higher-order gradients (default is True).
    :return: The gradient of `dy` with repect to `dx`.
    """

    if ones_like_tensor is None:
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(dy)]
    else:
        grad_outputs = ones_like_tensor

    if isinstance(dx, torch.Tensor):
        dx = [dx]

    dy_dx = torch.autograd.grad(
        outputs=[dy],
        inputs=dx,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=False,
    )

    grads = [
        grad if grad is not None else torch.zeros_like(dx[i])
        for i, grad in enumerate(dy_dx)
    ]
    return grads


if __name__ == "__main__":

    # BUG: the following function does not work properly.
    """
    def pde_loss(x, y, preds, extra_variables):
        u_pred, v_pred, p_pred = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]
        lambda_1, lambda_2 = 1.0, 0.02

        u_x, u_y = gradient(u_pred, [x, y])
        v_x, v_y = gradient(v_pred, [x, y])
        p_x, p_y = gradient(p_pred, [x, y])

        u_xx = gradient(u_x, x)[0]
        u_yy = gradient(u_y, y)[0]
        v_xx = gradient(v_x, x)[0]
        v_yy = gradient(v_y, y)[0]

        f_mass = u_x + v_y
        f_u = lambda_1 * (u_pred * u_x + v_pred * u_y) + p_x - lambda_2 * (u_xx + u_yy)
        f_v = lambda_1 * (u_pred * v_x + v_pred * v_y) + p_y - lambda_2 * (v_xx + v_yy)

        f_u_loss = F.mse_loss(f_u, torch.zeros_like(f_u))
        f_v_loss = F.mse_loss(f_v, torch.zeros_like(f_v))
        f_loss = F.mse_loss(f_mass, torch.zeros_like(f_mass))
        return f_u_loss + f_v_loss + f_loss
    """

    pass
