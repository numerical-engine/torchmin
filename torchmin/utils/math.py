import torch

import sys

def get_f_and_grad(func:callable, x:torch.Tensor, target_indices:tuple[int], detach:bool = True)->tuple[torch.float, torch.Tensor]:
    if target_indices is None:
        target_indices = tuple(range(x.shape[0]))

    f = func(x)
    grad_f = torch.autograd.grad(f, x)[0][target_indices,]

    if detach:
        return f.detach().clone(), grad_f.detach().clone()
    else:
        return f, grad_f

def _get_b(bounds:tuple[tuple[float, float]], x:torch.Tensor, target_indices:tuple[int], detach:bool = True)->torch.Tensor:
    constr_vector = []
    for i, bound in enumerate(bounds):
        if bound is not None:
            x_min, x_max = bound
            if x[i] < x_min:
                constr_vector.append(x_min - x[i])
            elif x[i] > x_max:
                constr_vector.append(x[i] - x_max)

    if len(constr_vector) == 0:
        return None
    else:
        constr_vector = torch.stack(constr_vector)

        if detach:
            return constr_vector.detach().clone()
        else:
            return constr_vector

def _get_b_and_matrix(bounds:tuple[tuple[float, float]], x:torch.Tensor, target_indices:tuple[int], detach:bool = True)->tuple[torch.Tensor, torch.Tensor]:
    def _tune_bounds(_bounds:tuple[tuple[float, float]], target_indices:tuple[int], dim:int)->tuple[tuple[float, float], ...]:
        bounds = []
        for idx, i in enumerate(range(dim)):
            if i in target_indices:
                bounds.append(_bounds[idx])
            else:
                bounds.append(None)
        return tuple(bounds)

    bounds = _tune_bounds(_bounds=bounds, target_indices=target_indices, dim=x.shape[0])
    constr_vector = []
    grad_constr_matrix = []

    with torch.no_grad():
        for i, bound in enumerate(bounds):
            if bound is not None:
                x_min, x_max = bound
                if x[i] < x_min:
                    constr_vector.append(x_min - x[i])
                    gcm = torch.zeros_like(x)
                    gcm[i] = -1.
                    grad_constr_matrix.append(gcm[target_indices,])
                elif x[i] > x_max:
                    constr_vector.append(x[i] - x_max)
                    gcm = torch.zeros_like(x)
                    gcm[i] = 1.
                    grad_constr_matrix.append(gcm[target_indices,])

    if len(constr_vector) == 0:
        return None, None
    else:
        grad_constr_matrix = torch.stack(grad_constr_matrix, dim = 1)
        constr_vector = torch.stack(constr_vector)

        if detach:
            return constr_vector.detach().clone(), grad_constr_matrix.detach().clone()
        else:
            return constr_vector, grad_constr_matrix


def _get_ceq(constr_eq:list[callable], x:torch.Tensor, target_indices:tuple[int], detach:bool = True)->torch.Tensor:
    constr_vector = []
    for c_eq in constr_eq:
        if detach:
            c_value, _ = get_f_and_grad(func=c_eq, x=x, target_indices=target_indices, detach=True)
        else:
            c_value = c_eq(x)
        constr_vector.append(c_value)

    if len(constr_vector) == 0:
        return None
    else:
        constr_vector = torch.stack(constr_vector)

        if detach:
            return constr_vector.detach().clone()
        else:
            return constr_vector


def _get_ceq_and_matrix(constr_eq:list[callable], x:torch.Tensor, target_indices:tuple[int], detach:bool = True)->tuple[torch.Tensor, torch.Tensor]:
    constr_vector = []
    grad_constr_matrix = []

    for c_eq in constr_eq:
        c_value, grad_c_eq = get_f_and_grad(func=c_eq, x=x, target_indices=target_indices, detach = detach)
        grad_constr_matrix.append(grad_c_eq)
        constr_vector.append(c_value)

    if len(constr_vector) == 0:
        return None, None
    else:
        grad_constr_matrix = torch.stack(grad_constr_matrix, dim = 1)
        constr_vector = torch.stack(constr_vector)

        if detach:
            return constr_vector.detach().clone(), grad_constr_matrix.detach().clone()
        else:
            return constr_vector, grad_constr_matrix


def _get_cl(constr_l:list[callable], x:torch.Tensor, target_indices:tuple[int], detach:bool = True)->torch.Tensor:
    constr_vector = []
    for c_l in constr_l:
        if detach:
            c_value, _ = get_f_and_grad(func=c_l, x=x, target_indices=target_indices, detach=True)
        else:
            c_value = c_l(x)
        if c_value > 0:
            constr_vector.append(c_value)

    if len(constr_vector) == 0:
        return None
    else:
        constr_vector = torch.stack(constr_vector)

        if detach:
            return constr_vector.detach().clone()
        else:
            return constr_vector


def _get_cl_and_matrix(constr_l:list[callable], x:torch.Tensor, target_indices:tuple[int], detach:bool = True, active_tol:float = 1e-8)->tuple[torch.Tensor, torch.Tensor]:
    constr_vector = []
    grad_constr_matrix = []

    for c_l in constr_l:
        c_value, grad_c_l = get_f_and_grad(func=c_l, x=x, target_indices=target_indices, detach = detach)
        if c_value > -active_tol:
            grad_constr_matrix.append(grad_c_l)
            constr_vector.append(c_value)

    if len(constr_vector) == 0:
        return None, None
    else:
        grad_constr_matrix = torch.stack(grad_constr_matrix, dim = 1)
        constr_vector = torch.stack(constr_vector)

        if detach:
            return constr_vector.detach().clone(), grad_constr_matrix.detach().clone()
        else:
            return constr_vector, grad_constr_matrix


def update_hessianinv_bfgs(H_current:torch.Tensor, x_current:torch.Tensor, x_before:torch.Tensor, grad_current:torch.Tensor, grad_before:torch.Tensor, target_indices:tuple[int], eps:float = 1e-12)->torch.Tensor:
    with torch.no_grad():
        s = x_current[target_indices,] - x_before[target_indices,]
        y = grad_current - grad_before
        rho = 1. / (torch.dot(y, s) + eps)
        H_next = (torch.eye(len(target_indices)) - rho * torch.outer(s, y)) @ H_current @ (torch.eye(len(target_indices)) - rho * torch.outer(y, s)) + rho * torch.outer(s, s)

    return H_next.detach().clone()

def update_hessianinv_bfgs_powell_damping(
    H_current:torch.Tensor, x_current:torch.Tensor, x_before:torch.Tensor, grad_current:torch.Tensor, grad_before:torch.Tensor, target_indices:tuple[int], eps:float = 1e-12, omega:float = 0.2)->torch.Tensor:

    with torch.no_grad():
        s = x_current[target_indices,] - x_before[target_indices,]
        y = grad_current - grad_before
        if torch.dot(s, y) < omega * torch.dot(s, H_current @ s):
            theta = (1. - omega) * torch.dot(s, H_current @ s) / (torch.dot(s, H_current @ s) - torch.dot(s, y) + eps)
            y = theta * y + (1 - theta) * H_current @ s
        rho = 1. / (torch.dot(y, s) + eps)
        H_next = (torch.eye(len(target_indices)) - rho * torch.outer(s, y)) @ H_current @ (torch.eye(len(target_indices)) - rho * torch.outer(y, s)) + rho * torch.outer(s, s)
    
        return H_next.detach().clone()


def update_hessian_bfgs(B_current:torch.Tensor, x_current:torch.Tensor, x_before:torch.Tensor, grad_current:torch.Tensor, grad_before:torch.Tensor, target_indices:tuple[int], eps:float = 1e-12)->torch.Tensor:
    with torch.no_grad():
        s = x_current[target_indices,] - x_before[target_indices,]
        y = grad_current - grad_before
        Bs = B_current @ s
        s_dot_Bs = torch.dot(s, Bs)
        y_dot_s = torch.dot(y, s)

        B_next = B_current - torch.outer(Bs, Bs) / (s_dot_Bs + eps) + torch.outer(y, y) / (y_dot_s + eps)

    return B_next.detach().clone()


def update_hessian_bfgs_powell_damping(
    B_current:torch.Tensor, x_current:torch.Tensor, x_before:torch.Tensor, grad_current:torch.Tensor, grad_before:torch.Tensor, target_indices:tuple[int], eps:float = 1e-12, omega:float = 0.2)->torch.Tensor:

    with torch.no_grad():
        s = x_current[target_indices,] - x_before[target_indices,]
        y = grad_current - grad_before
        s_dot_Bs = torch.dot(s, B_current @ s)
        y_dot_s = torch.dot(y, s)

        if y_dot_s < omega * s_dot_Bs:
            theta = (1. - omega) * s_dot_Bs / (s_dot_Bs - y_dot_s + eps)
            y = theta * y + (1 - theta) * (B_current @ s)

        Bs = B_current @ s
        s_dot_Bs = torch.dot(s, Bs)
        y_dot_s = torch.dot(y, s)

        B_next = B_current - torch.outer(Bs, Bs) / (s_dot_Bs + eps) + torch.outer(y, y) / (y_dot_s + eps)

    return B_next.detach().clone()