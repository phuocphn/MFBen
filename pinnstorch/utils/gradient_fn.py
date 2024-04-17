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
        [dy],
        dx,
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
