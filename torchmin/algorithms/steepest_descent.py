import torch

from torchmin.utils.math import get_f_and_grad

def sd(
    func:callable,
    x_init:torch.Tensor,
    line_search:"LineSearch",
    max_itr:int,
    xrel_tol:float = 1e-6,
    target_indices:tuple[int] = None,)->torch.Tensor:
    if target_indices is None:
        target_indices = tuple(range(x_init.shape[0]))

    x = x_init.detach().clone().requires_grad_(True)

    for _ in range(max_itr):
        value, grad = get_f_and_grad(func=func, x=x, target_indices=target_indices)
        alpha = line_search(func=func, x=x, value=value, grad=grad, target_indices=target_indices, d=-grad)
        with torch.no_grad():
            x[target_indices,] = x[target_indices,] - alpha * grad
            if torch.norm(alpha * grad) < xrel_tol*torch.norm(x):
                return x.detach().clone()
        x = x.detach().clone().requires_grad_(True)
        
    return x.detach().clone()