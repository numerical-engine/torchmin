import torch

from torchmin.utils import math

def get_step(
    func:callable,
    x:torch.Tensor,
    Hess:torch.Tensor,
    bounds:tuple[tuple[float, float]],
    line_search:"LineSearch",
    rho:float,
    constr_eq:list[callable],
    constr_lower:list[callable],
    target_indices:tuple[int])->tuple[torch.Tensor, float]:

    def func_w_penalty(x:torch.Tensor)->torch.float:
        c_b = math._get_b(bounds, x, target_indices, detach=False)
        c_eq = math._get_ceq(constr_eq = constr_eq, x = x, target_indices = target_indices, detach=False)
        c_l = math._get_cl(constr_l=constr_lower, x=x, target_indices=target_indices, detach=False)

        f = func(x)
        if c_b is not None:
            for c in c_b:
                f = f + rho * torch.max(torch.tensor(0., dtype=x.dtype, device=x.device), c)
        if c_eq is not None:
            f = f + rho * torch.sum(torch.abs(c_eq))
        if c_l is not None:
            for c in c_l:
                f = f + rho * torch.max(torch.tensor(0., dtype=x.dtype, device=x.device), c)

        return f
    
    constr_vector = []
    grad_constr_matrix = []
    c_b, grad_b = math._get_b_and_matrix(bounds, x, target_indices)
    if c_b is not None:
        constr_vector.append(c_b)
        grad_constr_matrix.append(grad_b)

    c_eq, grad_eq = math._get_ceq_and_matrix(constr_eq = constr_eq, x = x, target_indices = target_indices)
    if c_eq is not None:
        constr_vector.append(c_eq)
        grad_constr_matrix.append(grad_eq)

    c_l, grad_l = math._get_cl_and_matrix(constr_l=constr_lower, x=x, target_indices=target_indices)
    if c_l is not None:
        constr_vector.append(c_l)
        grad_constr_matrix.append(grad_l)

    constr_vector = torch.cat(constr_vector, dim = 0) if len(constr_vector) > 0 else None
    grad_constr_matrix = torch.cat(grad_constr_matrix, dim = 1) if len(grad_constr_matrix) > 0 else None

    _, grad = math.get_f_and_grad(func=func, x=x, target_indices=target_indices)

    if grad_constr_matrix is None:
        A = Hess.detach().clone()
        b = -grad
    else:
        A = torch.zeros(len(Hess) + grad_constr_matrix.shape[1], len(Hess) + grad_constr_matrix.shape[1], dtype=x.dtype, device=x.device)
        A[:len(Hess), :len(Hess)] = Hess
        A[:len(Hess), len(Hess):] = grad_constr_matrix
        A[len(Hess):, :len(Hess)] = grad_constr_matrix.T
        b = torch.cat([-grad, -constr_vector], dim=0)

    d = torch.linalg.solve(A, b)[:len(Hess)]

    x = x.detach().clone().requires_grad_(True)
    p_value, p_grad = math.get_f_and_grad(func=func_w_penalty, x=x, target_indices=target_indices)
    alpha = line_search(func=func_w_penalty, x=x, value=p_value, grad=p_grad, target_indices=target_indices, d=d)

    return d*alpha




def sqp(
    func:callable,
    x_init:torch.Tensor,
    bounds:tuple[tuple[float, float]],
    line_search:"LineSearch",
    max_itr:int,
    rho:float,
    constr_eq:list[callable] = (),
    constr_lower:list[callable] = (),
    xrel_tol:float = 1e-6,
    hessian_calc:str = "powell_damping",
    eps:float = 1e-12,
    omega:float = 0.2,
    target_indices:tuple[int] = None,)->torch.Tensor:

    if target_indices is None:
            target_indices = tuple(range(x_init.shape[0]))

    x = x_init.detach().clone().requires_grad_(True)
    Hess = torch.eye(len(target_indices), dtype=x.dtype, device=x.device)

    for _ in range(max_itr):
        _, grad = math.get_f_and_grad(func=func, x=x, target_indices=target_indices)
        dx = get_step(func=func, x=x, Hess = Hess, bounds=bounds, line_search=line_search, rho=rho, constr_eq=constr_eq, constr_lower=constr_lower, target_indices=target_indices)
        x_next = x.detach().clone()
        with torch.no_grad():
            x_next[target_indices,] = x[target_indices,] + dx
            if torch.norm(dx) < xrel_tol*torch.norm(x):
                return x_next.detach().clone()

        x_next.requires_grad_(True)
        _, grad_next = math.get_f_and_grad(func=func, x=x_next, target_indices=target_indices)

        if hessian_calc == "simple":
            Hess = math.update_hessian_bfgs(Hess, x_next, x, grad_next, grad, target_indices, eps)
        elif hessian_calc == "powell_damping":
            Hess = math.update_hessian_bfgs_powell_damping(Hess, x_next, x, grad_next, grad, target_indices, eps, omega)
        else:
            raise ValueError(f"hessian_calc must be either 'simple' or 'powell_damping', but got {hessian_calc}")

        x = x_next.detach().clone().requires_grad_(True)

    return x.detach().clone()