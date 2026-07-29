import torch

from torchmin.utils.math import get_f_and_grad, update_hessianinv_bfgs, update_hessianinv_bfgs_powell_damping

def bfgs(
    func:callable,
    x_init:torch.Tensor,
    line_search:"LineSearch",
    max_itr:int,
    xrel_tol:float = 1e-6,
    hessian_calc:str = "powell_damping",
    eps:float = 1e-12,
    omega:float = 0.2,
    target_indices:tuple[int] = None,)->torch.Tensor:

    if target_indices is None:
        target_indices = tuple(range(x_init.shape[0]))

    x = x_init.detach().clone().requires_grad_(True)

    Hinv = torch.eye(len(target_indices), dtype=x.dtype, device=x.device)

    for _ in range(max_itr):
        value, grad = get_f_and_grad(func=func, x=x, target_indices=target_indices)
        d = -Hinv @ grad
        alpha = line_search(func=func, x=x, value=value, grad=grad, target_indices=target_indices, d=d)

        x_next = x.detach().clone()
        with torch.no_grad():
            x_next[target_indices,] = x[target_indices,] + alpha * d
            if torch.norm(alpha * d) < xrel_tol*torch.norm(x):
                return x_next.detach().clone()

        x_next.requires_grad_(True)

        _, grad_next = get_f_and_grad(func, x_next, target_indices)

        if hessian_calc == "simple":
            Hinv = update_hessianinv_bfgs(Hinv, x_next, x, grad_next, grad, target_indices, eps)
        elif hessian_calc == "powell_damping":
            Hinv = update_hessianinv_bfgs_powell_damping(Hinv, x_next, x, grad_next, grad, target_indices, eps, omega)
        else:
            raise ValueError(f"hessian_calc must be either 'simple' or 'powell_damping', but got {hessian_calc}")

        x = x_next.detach().clone().requires_grad_(True)

    return x.detach().clone()