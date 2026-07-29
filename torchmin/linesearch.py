import torch

class AbstClass:
    def __call__(self, func:callable, x:torch.Tensor, value:torch.foat, grad:torch.Tensor, target_indices:tuple[int] = None, d:torch.Tensor = None)->float:
        if target_indices is None:
                    target_indices = tuple(range(x.shape[0]))
        
        if d is None:
            d = -grad.clone()
        assert d.shape == grad.shape, f"d must have shape {grad.shape}, but got {d.shape}"
        assert d.shape == (len(target_indices),), f"d must have shape ({len(target_indices)},), but got {d.shape}"

        return self.forward(func, x, value, grad, target_indices, d)

    def forward(self, func:callable, x:torch.Tensor, value:torch.foat, grad:torch.Tensor, target_indices:tuple[int], d:torch.Tensor)->float:
        raise NotImplementedError("Subclasses must implement the forward method.")



class LineArmijo(AbstClass):
    def __init__(self, tau:float = 0.5, alpha:float = 1.0, sigma:float = 0.5, alpha_min:float = 0.)->None:
        assert 0 < sigma < 1, "sigma must be in (0, 1)"
        assert 0 < tau < 1, "tau must be in (0, 1)"
        assert alpha > alpha_min, f"alpha must be greater than alpha_min={alpha_min}, but got alpha={alpha}"
        assert alpha_min >= 0., f"alpha_min must be greater than 0, but got alpha_min={alpha_min}"

        self.tau = tau
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_min = alpha_min

    def forward(self, func:callable, x:torch.Tensor, value:torch.foat, grad:torch.Tensor, target_indices:tuple[int], d:torch.Tensor)->float:
        """バックトラッキング法によるArmijo条件を満たす実数alphaを出力する。

        Args:
            func (callable): 目的関数。出力はスカラー値である必要がある。
            x (torch.Tensor): 現時点の候補解。shapeは(変数の次元数,)である必要がある。
            value (torch.float): 現時点の候補解xにおける目的関数の値。
            grad (torch.Tensor): 現時点の候補解xにおける目的関数の勾配。shapeは(len(target_indices),)である必要がある。
            target_indices (tuple[int]): 入力のうち更新対象のインデックスのタプル。例えば、xが3次元でtarget_indices=(0, 2)とすると、x[0]とx[2]のみ最適化対象。
            d (torch.Tensor, optional): 降下方向。shapeは(len(target_indices),)である必要がある。
        
        Returns:
            float: Armijo条件を満たすステップサイズalpha。alpha_minより大きいalphaで条件を満たすものが見つからなかった場合はalpha_minを返す。
        """

        with torch.no_grad():
            grad_dot_d = torch.dot(grad, d)
            alpha = self.alpha
            while alpha > self.alpha_min:
                x_new = x.clone()
                x_new[target_indices,] = x[target_indices,] + alpha * d
                if func(x_new) <= value + self.sigma * alpha * grad_dot_d:
                    return alpha
                alpha *= self.tau

            return self.alpha_min


class LineWolfe(AbstClass):
    def __init__(self, tau:float = 0.5, alpha:float = 1.0, sigma:float = 0.5, beta:float = 0.9, alpha_min:float = 0.)->None:
        assert 0 < sigma < beta < 1, "sigma and beta must satisfy 0 < sigma < beta < 1"
        assert 0 < tau < 1, "tau must be in (0, 1)"
        assert alpha > alpha_min, f"alpha must be greater than alpha_min={alpha_min}, but got alpha={alpha}"
        assert alpha_min >= 0., f"alpha_min must be greater than 0, but got alpha_min={alpha_min}"

        self.tau = tau
        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta
        self.alpha_min = alpha_min

    def forward(self, func:callable, x:torch.Tensor, value:torch.foat, grad:torch.Tensor, target_indices:tuple[int], d:torch.Tensor)->float:
        grad_dot_d = torch.dot(grad, d)

        alpha = self.alpha
        while alpha > self.alpha_min:
            x_new = x.detach().clone()
            x_new[target_indices,] = x[target_indices,] + alpha * d
            x_new.requires_grad_(True)

            value_new = func(x_new)
            grad_new = torch.autograd.grad(value_new, x_new)[0][target_indices,]

            if value_new <= value + self.sigma * alpha * grad_dot_d and torch.dot(grad_new, d) >= self.beta * grad_dot_d:
                return alpha

            alpha *= self.tau

        return self.alpha_min


class LineWolfeStrong(LineWolfe):
    def forward(self, func:callable, x:torch.Tensor, value:torch.foat, grad:torch.Tensor, target_indices:tuple[int], d:torch.Tensor)->float:
        grad_dot_d = torch.dot(grad, d)
        abs_grad_dot_d = torch.abs(grad_dot_d)

        alpha = self.alpha
        while alpha > self.alpha_min:
            x_new = x.detach().clone()
            x_new[target_indices,] = x[target_indices,] + alpha * d
            x_new.requires_grad_(True)

            value_new = func(x_new)
            grad_new = torch.autograd.grad(value_new, x_new)[0][target_indices,]

            if value_new <= value + self.sigma * alpha * grad_dot_d and torch.abs(torch.dot(grad_new, d)) <= self.beta * abs_grad_dot_d:
                return alpha

            alpha *= self.tau

        return self.alpha_min