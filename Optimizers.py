import time

import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import List
import copy
from Dataset import Dataset
from Model import FNN
from functools import *

class CMA(Optimizer):
    def __init__(self, params, alpha=1e-3, zeta=1e-3, eps=1e-3,theta=1e-3,
                 delta=1e-3,gamma=1e-3,tau=1e-2,verbose=False,max_it_EDFL=100,
                 verbose_EDFL=False):


        defaults = dict(alpha=alpha, zeta=zeta, eps=eps,theta=theta,
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
             mod: FNN,
             dataset: Dataset,
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
        with torch.no_grad():
            idx = 0
            for param in sample_model.parameters():
                param.copy_(w_prova[idx:idx + param.numel()].reshape(param.shape))

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
            with torch.no_grad():
                idx = 0
                for param in sample_model.parameters():
                    param.copy_(w_prova[idx:idx + param.numel()].reshape(param.shape))
            cur_loss = closure(dataset,device,sample_model,criterion)
            nfev += 1
            idx += 1
        return alpha, nfev, f_ad


    def control_step(self,
                     model: FNN,
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
        f_before = self.f_before
        fw0 = self.fw0
        w_after = model.get_w()
        d = (w_after - w_before) / zeta  # Descent direction d_tilde
        f_tilde = closure(dataset, device, model, criterion)
        history['nfev'] += 1


        if f_tilde < f_before - gamma * zeta:  # This is the best case, Exit at step 6
            f_after = f_tilde  # The value of the objective function is f_tilde
            history['accepted'].append(epoch)
            history['Exit'].append('6')
            if verbose: print('ok inner cycle')

        else:
            # Go back to the previous value and check... Maybe we can still do something. Step 7
            model.set_w(w_before)
            if verbose: print('back to w_k')

            if torch.linalg.norm(d) <= tau * zeta:  # Step 8, we check ||d||
                if verbose: print('||d|| suff piccola  -->  Step size reduced')
                self.set_zeta(zeta * theta)  # Reduce step size, Step 9
                if f_tilde <= fw0:
                    alpha = zeta
                    new_w = w_before + alpha * d
                    model.set_w(new_w)
                    f_after = f_tilde  # Exit 9a, the tentative point is accepted  after an
                    # additional control on ||d|| but the step-size is reduced
                    history['Exit'].append('9a')
                else:
                    alpha = 0  # Exit 9b. We are no more in the level set, we cannot accept w_tilde
                    f_after = f_before
                    history['Exit'].append('9b')

            else:  # Step 10, d_tilde not too small, we perform EDFL
                if verbose: print('Executing EDFL')
                alpha, nf_EDFL, f_after_LS = self.EDFL(model, dataset, w_before, f_before,
                                                            f_tilde, d, closure, device, criterion)
                history['nfev'] += nf_EDFL
                if alpha * torch.linalg.norm(d) ** 2 <= tau * zeta:  # Step 12, as in Step 8
                    zeta = zeta * theta  # Reduce the step size
                    self.set_zeta(zeta)
                    if alpha > 0:
                        if verbose: print('LS accepted')  # Step 13a executed
                        f_after = f_after_LS
                        history['Exit'].append('13a')
                        pass
                    elif alpha == 0 and f_tilde <= fw0:
                        if verbose: print('Step reduced but w_tilde accepted')  # Step 13b
                        alpha = zeta
                        f_after = f_tilde
                        history['Exit'].append('13b')
                    else:
                        if verbose: print('Total fail')  # Step 13c
                        alpha = 0
                        f_after = f_before
                        history['Exit'].append('13c')
                else:  # Perform step 15, the LS is a total success, we accept alpha and do not reduce zeta
                    f_after = f_after_LS
                    history['Exit'].append('15')

            # We set w_k+1 = w + alpha*d
            if verbose: print(f' Final alpha = {alpha}   Current step-size zeta =  {zeta}')
            if alpha > 0:  # If alpha is not zero, set the variables to the new value
                new_w = w_before + alpha * d
                model.set_w(new_w)
                history['nfev'] += 1

        if verbose: print(f'f_before: {f_before:3e} f_tilde: {f_tilde:3e} f_after: {f_after:3e}  Exit Step: {history["Exit"][-1]}')
        history['step_size'].append(self.param_groups[0]['zeta'])
        if verbose: print(f'Step-size: {self.param_groups[0]["zeta"]:3e}')
        return model, history, f_before, f_after, history['Exit'][-1]



class NMCMA(Optimizer):
    def __init__(self, params, alpha=1e-3, zeta=1e-3, eps=1e-3,theta=1e-3,
                 delta=1e-3,gamma=1e-3,tau=1e-2,verbose=False,max_it_EDFL=100,
                 verbose_EDFL=False, M = 5):


        defaults = dict(alpha=alpha, zeta=zeta,
        eps=eps,theta=theta,delta=delta,
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
             mod: FNN,
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
                     model: FNN,
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
        w_after = model.get_w()
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
            model.set_w(w_before)
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
                model.set_w(new_w)

        if verbose: print(f' Rk {R_k:3e}   f_before: {f_before:3e} f_tilde: {f_tilde:3e} f_after: {f_after:3e}  Exit: {history["Exit"][-1]}')
        if verbose: print(f' Step-size: {self.param_groups[0]["zeta"]:3e}')
        history['step_size'].append(self.param_groups[0]['zeta'])
        return model, history, f_before, f_after, history['Exit'][-1]


class IG(Optimizer):

    def __init__(self, params, zeta=1e-3, eps=1e-3, verbose=False):
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






def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.



def _strong_wolfe(obj_func,
                  x,
                  t,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9,
                  tolerance_change=1e-9,
                  max_ls=25):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals



class LBFGS(Optimizer):
    """Implements L-BFGS algorithm, heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.

    ... warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    ... warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    ... note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Args:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    def __init__(self,
                 params,
                 lr=1,
                 max_iter=1000000000,
                 max_eval=1000000000,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-9,
                 history_size=100,
                 line_search_fn='strong_wolfe',
                 time_limit = 100,
                 verbose = False):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
            time_limit = time_limit,
            verbose = verbose)
        super(LBFGS, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d,*args, **kwargs):
        self._add_grad(t, d)
        loss = float(closure(*args, **kwargs))
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def step(self, closure, *args, **kwargs):
        """Performs a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']
        time_limit = group['time_limit']
        verbose = group['verbose']

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure(*args,**kwargs)
        loss = float(orig_loss)
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        n_iter = 0
        # optimize for a max of max_iter iterations
        elapsed_time = 0
        while n_iter < max_iter and elapsed_time <= time_limit:
            # keep track of nb of iterations
            if verbose: print(f'It. {n_iter+1}  Time: {elapsed_time:.2f}')
            n_iter += 1
            state['n_iter'] += 1
            t_start = time.time()

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state['n_iter'] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if 'al' not in state:
                    state['al'] = [None] * history_size
                al = state['al']

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(old_dirs[i], alpha=-al[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(old_stps[i], alpha=al[i] - be_i)

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state['n_iter'] == 1:
                t = min(1., 1. / flat_grad.abs().sum()) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn != "strong_wolfe":
                    raise RuntimeError("only 'strong_wolfe' is supported")
                else:
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d,*args, **kwargs)

                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func, x_init, t, d, loss, flat_grad, gtd)
                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    with torch.enable_grad():
                        loss = float(closure(*args, **kwargs))
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                if verbose: print('Max evals')
                break

            # optimal condition
            if opt_cond:
                if verbose: print('Optimal cond')
                break

            # lack of progress
            if d.mul(t).abs().max() <= tolerance_change:
                if verbose: print('Optimal cond 1')
                break

            if abs(loss - prev_loss) < tolerance_change:
                if verbose: print('Optimal cond 2')
                break
            iteration_time = time.time()-t_start
            elapsed_time += iteration_time

        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        return orig_loss
