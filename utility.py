import numpy as np
import pickle
import pandas as pd
from Model import *
from Optimizers import *
from Dataset import Dataset



def closure(dataset: Dataset,
            device: torch.device,
            mod,
            loss_fun,
            test = False,
            rho = 1e-6): #Used to compute the loss on the entire data set
    with torch.no_grad():
        try:
            if test == False:
                X = dataset.x_train.to(device)
                Y = dataset.y_train.to(device)
            else:
                X = dataset.x_test.to(device)
                Y = dataset.y_test.to(device)
            Y_pred = mod(X)
            loss = loss_fun(Y,Y_pred)
        except:
            loss = closure_fragmented(dataset,device,loss_fun,test)
        reg_term = 0
        for param in mod.parameters():
            reg_term += torch.linalg.norm(param)**2

    return loss + rho*reg_term


def closure_fragmented(dataset: Dataset,
                device: torch.device,
                mod,
                loss_fun,
                test=False,
                rho = 1e-6): #Used in case of GPU/CPU overflow
    with torch.no_grad():
        if test == False:
            mb_size = 128
            n_it = np.ceil((dataset.P/mb_size))
            L = 0
            for j in range(n_it):
                dataset.minibatch(j*mb_size,(j+1)*mb_size)
                x = dataset.x_train_mb.to(device)
                y = dataset.y_train_mb.to(device)
                y_pred = mod(x)
                loss = loss_fun(y,y_pred).item()
                L += (len(x)/dataset.P)*loss
        else:
            L = -100
        reg_term = 0
        for param in mod.parameters():
            reg_term += torch.linalg.norm(param)**2
    return L + rho*reg_term





def save_csv_history(path,ID):
    objects = []
    with (open(path + '.pkl', "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    df = pd.DataFrame(objects)
    df.to_csv(path + ID +'.csv', header=False, index=False, sep=" ")


def set_architecture(arch:str, input_dim:int, seed: int):
    torch.manual_seed(seed)
    if arch=='S':
        return nn.Sequential(nn.Linear(input_dim,50),nn.Sigmoid(),nn.Linear(50,1))
    elif arch=='M':
        return nn.Sequential(nn.Linear(input_dim,20),nn.Sigmoid(),nn.Linear(20,20),
                             nn.Sigmoid(), nn.Linear(20,20), nn.Sigmoid(), nn.Linear(20,1))
    elif arch=='L':
        return nn.Sequential(nn.Linear(input_dim,50),nn.Sigmoid(),nn.Linear(50,50),
                             nn.Sigmoid(),nn.Linear(50,50),nn.Sigmoid(),nn.Linear(50,50),
                             nn.Sigmoid(),nn.Linear(50,50), nn.Sigmoid(), nn.Linear(50,1))
    elif arch == 'XL':
        return nn.Sequential(nn.Linear(input_dim, 50), nn.Sigmoid(), nn.Linear(50, 50),
                             nn.Sigmoid(), nn.Linear(50, 50), nn.Sigmoid(), nn.Linear(50, 50),
                             nn.Sigmoid(), nn.Linear(50, 50), nn.Sigmoid(),
                             nn.Linear(50, 50), nn.Sigmoid(), nn.Linear(50, 50), nn.Sigmoid(),
                             nn.Linear(50, 50), nn.Sigmoid(), nn.Linear(50, 50), nn.Sigmoid(),
                             nn.Linear(50, 50), nn.Sigmoid(), nn.Linear(50, 1))
    elif arch == 'XXL':
        return nn.Sequential(nn.Linear(input_dim, 50), nn.Sigmoid(), nn.Linear(50, 300),
                             nn.Sigmoid(), nn.Linear(300, 300), nn.Sigmoid(), nn.Linear(300, 300),
                             nn.Sigmoid(), nn.Linear(300, 300),nn.Sigmoid(), nn.Linear(300, 300),
                             nn.Sigmoid(), nn.Linear(300, 300),nn.Sigmoid(), nn.Linear(300, 300),
                             nn.Sigmoid(), nn.Linear(300, 300),nn.Sigmoid(), nn.Linear(300, 300),
                             nn.Sigmoid(), nn.Linear(300, 300),nn.Sigmoid(), nn.Linear(300, 300),
                             nn.Sigmoid(), nn.Linear(300, 50),nn.Linear(50, 1))
    elif arch == 'XXXL':
        return nn.Sequential(nn.Linear(input_dim, 50), nn.Sigmoid(), nn.Linear(50, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 50),nn.Linear(50, 1))

    elif arch == '4XL':
        return nn.Sequential(nn.Linear(input_dim, 50), nn.Sigmoid(), nn.Linear(50, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 50),nn.Linear(50, 1))
    else:
        raise SystemError('Set an architecture in {S,M,L,XL,XXL,XXXL,4XL} and try again')


def set_optimizer(opt: str, model: FNN, *args, **kwargs):
    if opt == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), *args, **kwargs)
    elif opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), *args, **kwargs)
    elif opt == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), *args, **kwargs)
    elif opt == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters(), *args, **kwargs)
    elif opt == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(), *args, **kwargs)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), *args, **kwargs)
    elif opt == 'cma':
        optimizer = CMA(model.parameters(), *args, **kwargs)
    elif opt == 'lbfgs':
        optimizer = LBFGS(model.parameters(), *args, **kwargs)
    elif opt == 'ig':
        optimizer = IG(model.parameters(), *args, **kwargs)
    elif opt == 'nmcma':
        optimizer = NMCMA(model.parameters(), *args, **kwargs)
    else:
        raise SystemError('Optimizer not supported')

    return optimizer


