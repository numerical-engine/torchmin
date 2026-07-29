import torch
import torchmin

def rosenbrock(x:torch.Tensor)->torch.float:
    return (1. - x[0])**2 + 100.*(x[1] - x[0]**2)**2

x_init = torch.tensor([-1.2, 1.], dtype=torch.float)
line_search = torchmin.linesearch.LineArmijo(tau=0.5, alpha=1., sigma=0.5, alpha_min=1e-8)
x = torchmin.algorithms.bfgs(rosenbrock, x_init, line_search, 1000)
print(x)