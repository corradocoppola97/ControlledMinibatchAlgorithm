import torch
from torch.optim import Optimizer
from typing import Union
import copy
from Dataset import Dataset

class CMAFD(Optimizer):
    def __init__(self, params,
                 zeta=0.05,
                 theta=0.5,
                 delta=0.9,
                 gamma=1e-6,
                 tau=1e-2,
                 bar_zeta = 1e-2,
                 verbose=False,
                 max_it_EDFL=100,
                 verbose_EDFL=False):


        defaults = dict(zeta=zeta,theta=theta,bar_zeta=bar_zeta,
                    delta=delta,gamma=gamma,verbose=verbose,maximize=False,
                    tau=tau,max_it_EDFL=max_it_EDFL,verbose_EDFL=verbose_EDFL)

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('verbose', False)
            group.setdefault('maximize', False)

    def set_zeta(self,zeta):
        for group in self.param_groups:
            group['zeta'] = zeta

    def set_fw0(self,
                fw0: float):
        self.fw0 = fw0

    def set_reference(self,
                      f_before: float):
        self.f_before = f_before

    def step(self, closure=None,*args,**kwargs):
        loss = None
        if closure is not None:
            with torch.no_grad():
                loss = closure(*args,**kwargs)  #This should be used only when computing the loss on the whole data set
        else:
            with torch.no_grad():
                for group in self.param_groups:
                    for p in group['params']:
                        p.add_(p.grad, alpha=-group['zeta'])
        return loss

    def EDFL(self,
             mod,
             dataset: Union[Dataset,torch.utils.data.DataLoader],
             w_prima: torch.Tensor,
             loss_prima: float,
             loss_dopo: float,
             d_k: torch.Tensor,
             closure: callable,
             device: torch.device,
             criterion: torch.nn):

        zeta = self.param_groups[0]['zeta']
        gamma = self.defaults.get('gamma')
        delta = self.defaults.get('delta')
        verbose = self.defaults.get('verbose_EDFL')
        alpha = zeta
        nfev = 0
        if verbose: print(f'alpha inizio EDFL   {alpha}')
        sample_model = copy.deepcopy(mod)

        if loss_dopo > loss_prima - gamma * alpha * torch.linalg.norm(d_k) ** 2:
            if verbose: print('fail: ALPHA = 0')
            alpha = 0
            return alpha, 0, loss_dopo

        w_prova = w_prima + d_k * (alpha / delta)
        set_w(sample_model,w_prova)

        cur_loss = closure(dataset,device,sample_model,criterion)
        nfev += 1

        idx = 0
        f_ad = loss_dopo
        while cur_loss <= min(f_ad,loss_prima - gamma * (alpha / delta) * torch.linalg.norm(d_k) ** 2) \
                and idx <= self.defaults.get('max_it_EDFL'):
            if verbose: print('idx = ', idx)
            f_ad = cur_loss
            alpha = alpha / delta
            w_prova = w_prima + d_k * (alpha / delta)
            set_w(sample_model,w_prova)
            cur_loss = closure(dataset,device,sample_model,criterion)
            nfev += 1
            idx += 1
        return alpha, nfev, f_ad


    def control_step(self,
                     model,
                     w_before: torch.Tensor,
                     closure: callable,
                     dataset: Union[Dataset,torch.utils.data.DataLoader],
                     device: torch.device,
                     criterion: torch.nn,
                     history: dict,
                     epoch: int):

        zeta = self.param_groups[0]['zeta']
        bar_zeta = self.param_groups[0]['bar_zeta']
        verbose = self.param_groups[0]['verbose']
        f_before = self.f_before
        w_after = get_w(model)
        d = (w_after - w_before) / zeta  # Descent direction d_tilde
        f_tilde = closure(dataset, device, model, criterion)
        history['nfev'] += 1


        if f_tilde < f_before - bar_zeta:  # This is the best case, Exit at step 6
            f_after = f_tilde  # The value of the objective function is f_tilde
            history['accepted'].append(epoch)
            history['Exit'].append('6')
            if verbose: print('ok inner cycle')

        else:
            # Go back to the previous value and perform EDFL
            set_w(model,w_before)
            alpha, nf_EDFL, f_after_LS = self.EDFL(model, dataset, w_before, f_before,
                                                        f_tilde, d, closure, device, criterion)
            if alpha > 0:
                if verbose: print('LS accepted')  # Step 13a executed
                f_after = f_after_LS
                history['Exit'].append('12')
            else:
                if verbose: print('Total fail')  # Step 13c
                f_after = f_before
                history['Exit'].append('16')


            # We set w_k+1 = w + alpha*d
            if verbose: print(f' alpha_k = {alpha}   zeta_k =  {zeta}')
            if alpha > 0:  # If alpha is not zero, set the variables to the new value
                new_w = w_before + alpha * d
                set_w(model,new_w)
                history['nfev'] += 1

        if verbose: print(f'f_before: {f_before:3e} f_tilde: {f_tilde:3e} f_after: {f_after:3e}  Exit Step: {history["Exit"][-1]}')
        history['step_size'].append(self.param_groups[0]['zeta'])
        if verbose: print(f'Step-size: {self.param_groups[0]["zeta"]:3e}')
        return model, history, f_before, f_after, history['Exit'][-1]


def get_w(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_w(model, w):
    index = 0
    for param in model.parameters():
        param_size = torch.numel(param)
        param.data = w[index:index+param_size].view(param.size()).to(param.device)
        index += param_size