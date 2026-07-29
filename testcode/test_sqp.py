import torch
import torchmin

def HS71(x:torch.Tensor)->torch.float:
    return x[0]*x[3]*(x[0] + x[1] + x[2]) + x[2]

def const_l(x:torch.Tensor)->torch.float:
    return 25. - x[0]*x[1]*x[2]*x[3]

def const_eq(x:torch.Tensor)->torch.float:
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 - 40.


x_init = torch.tensor([1., 5., 5., 1.], dtype=torch.float)
line_search = torchmin.linesearch.LineArmijo(tau=0.5, alpha=1., sigma=0.5, alpha_min=1e-8)
bounds = ((1., 5.), (1., 5.), (1., 5.), (1., 5.))

x = torchmin.algorithms.sqp(HS71, x_init, bounds, line_search, 1000, 1.0, (const_eq, ), (const_l, ), target_indices=(1, 2, 3))
print(x)