import torch
from torch.optim import Optimizer
import copy
from Dataset import Dataset


class NMCMA(Optimizer):
    def __init__(self, params, zeta=0.05,theta=0.5,
                 delta=0.9,gamma=1e-6,tau=1e-2,verbose=False,max_it_EDFL=100,
                 verbose_EDFL=False, M = 5):


        defaults = dict(zeta=zeta,theta=theta,delta=delta,
        gamma=gamma,verbose=verbose,maximize=False,
        tau=tau,max_it_EDFL=max_it_EDFL,
        verbose_EDFL=verbose_EDFL,M=M)

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

    def set_Rk(self,
                R_k: float):
        self.R_k = R_k

    def set_f_before(self,
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


    def NMEDFL(self,
             mod,
             dataset: Dataset,
             w_prima: torch.Tensor,
             loss_dopo: float,
             d_k: torch.Tensor,
             closure: callable,
             device: torch.device,
             criterion: torch.nn,
             R_k: float):

        zeta = self.param_groups[0]['zeta']
        gamma = self.defaults.get('gamma')
        delta = self.defaults.get('delta')
        verbose = self.defaults.get('verbose_EDFL')
        alpha = zeta
        nfev = 0
        if verbose: print(f'alpha inizio EDFL   {alpha}')
        sample_model = copy.deepcopy(mod)

        if loss_dopo > R_k - gamma * (alpha**2) * torch.linalg.norm(d_k) ** 2:
            if verbose: print('fail: ALPHA = 0')
            alpha = 0
            return alpha, 0, loss_dopo

        w_prova = w_prima + d_k * (alpha / delta)
        with torch.no_grad():
            idx = 0
            for param in sample_model.parameters():
                param.copy_(w_prova[idx:idx + param.numel()].reshape(param.shape))

        cur_loss = closure(dataset,device,sample_model,criterion)
        nfev += 1

        idx = 0
        f_ad = loss_dopo
        while cur_loss <= min(f_ad,R_k - gamma * ((alpha / delta)**2) * torch.linalg.norm(d_k) ** 2) \
                and idx <= self.defaults.get('max_it_EDFL'):
            if verbose: print('idx = ',idx)
            f_ad = cur_loss
            alpha = alpha / delta
            w_prova = w_prima + d_k * (alpha / delta)
            with torch.no_grad():
                idx = 0
                for param in sample_model.parameters():
                    param.copy_(w_prova[idx:idx + param.numel()].reshape(param.shape))
            cur_loss = closure(dataset,device,sample_model,criterion)
            nfev += 1
            idx += 1
        return alpha, nfev, f_ad


    def control_step(self,
                     model,
                     w_before: torch.Tensor,
                     closure: callable,
                     dataset: Dataset,
                     device: torch.device,
                     criterion: torch.nn,
                     history: dict,
                     epoch: int):

        zeta = self.param_groups[0]['zeta']
        gamma = self.defaults.get('gamma')
        theta = self.defaults.get('theta')
        tau = self.defaults.get('tau')
        verbose = self.param_groups[0]['verbose']
        w_after = get_w(model)
        d = (w_after - w_before) / zeta  # Descent direction \tilde d
        f_tilde = closure(dataset, device, model, criterion)
        f_before = history['train_loss'][-1]
        history['nfev'] += 1
        R_k = self.R_k

        if f_tilde < R_k - gamma * max(zeta, zeta * torch.linalg.norm(d)):
            if verbose: print('ok inner cycle')  # Inner cycle accepted, Step 7 is executed
            f_after = f_tilde
            history['accepted'].append(epoch)
            history['Exit'].append('7')

        else:
            # Go back to the previous value... Let's try to do something
            set_w(model,w_before)
            if verbose: print('Back to w_k')

            # Check if ||d|| is sufficiently small
            if torch.linalg.norm(d) <= tau * zeta:
                if verbose: print('||d|| suff piccola')
                zeta = zeta * theta
                self.set_zeta(zeta)
                alpha = 0  # Step 10 is executed, zeta reduced, alpha driven to zero, FAIL
                f_after = f_before
                history['Exit'].append('10')


            else:  # If not, perform the EDFL
                if verbose: print('perform NMEDFL')
                alpha, nf_EDFL, f_after_LS = self.NMEDFL(model, dataset, w_before, f_tilde,
                                                              d, closure, device, criterion, R_k)
                if (alpha ** 2) * torch.linalg.norm(d) ** 2 <= tau * zeta:
                    zeta = zeta * theta
                    self.set_zeta(zeta)  # Reduce step-size and accept LS, Step 14 is executed
                    history['Exit'].append('14')
                    f_after = f_after_LS
                else:
                    # Everything is ok, accept alpha and do not update zeta, Step 16 is executed
                    history['Exit'].append('16')
                    f_after = f_after_LS

            # We set w_k+1 = w + alpha*d
            if verbose: print(f' alpha = {alpha}\n')
            if alpha > 0:
                new_w = w_before + alpha * d
                set_w(model,new_w)

        if verbose: print(f' Rk {R_k:3e}   f_before: {f_before:3e} f_tilde: {f_tilde:3e} f_after: {f_after:3e}  Exit: {history["Exit"][-1]}')
        if verbose: print(f' Step-size: {self.param_groups[0]["zeta"]:3e}')
        history['step_size'].append(self.param_groups[0]['zeta'])
        return model, history, f_before, f_after, history['Exit'][-1]


def get_w(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_w(model, w):
    index = 0
    for param in model.parameters():
        param_size = torch.numel(param)
        param.data = w[index:index+param_size].view(param.size()).to(param.device)
        index += param_size