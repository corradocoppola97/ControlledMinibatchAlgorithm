import torch
from torch.optim import Optimizer
from typing import List, Union
from torch import Tensor

class IG(Optimizer):

    def __init__(self, params, zeta=0.5, eps=1e-3, verbose=False):
        defaults = dict(zeta=zeta,verbose=verbose,maximize=False, eps=eps)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('verbose', False)
            group.setdefault('maximize', False)

    def _init_group(self, group, params_with_grad, d_p_list):
        has_sparse_grad = False

        for p in group['params']:
            has_sparse_grad = False
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True
                state = self.state[p]
        return has_sparse_grad


    def update_zeta(self):
        for group in self.param_groups:
            zeta = group['zeta']
            group['zeta'] = zeta*(1-zeta*group['eps'])


    def step(self,*args,**kwargs):
        loss = None
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            hsg = self._init_group(group,params_with_grad,d_p_list)

            inner_cycle(params_with_grad,
                d_p_list,
                zeta=group['zeta'],
                maximize=group['maximize'])

        return loss


def inner_cycle(params: List[Tensor],
        d_p_list: List[Tensor],
        *,
        zeta: float,
        maximize: bool):

    with torch.no_grad():
        for i, param in enumerate(params):
            d_p = d_p_list[i] if not maximize else -d_p_list[i]
            param.add_(d_p, alpha=-zeta)

def get_w(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_w(model, w):
    index = 0
    for param in model.parameters():
        param_size = torch.numel(param)
        param.data = w[index:index+param_size].view(param.size()).to(param.device)
        index += param_size