import random
import torch.optim.lbfgs
from Model import FNN
from Dataset import *
from utility import  *
import os, time
from warnings import filterwarnings
filterwarnings('ignore')



def train_model(ds: str,
                arch: str,
                sm_root: str,
                ep: int,
                time_limit: int,
                batch_size:int,
                seed: int,
                ID_history = '',
                verbose_train = False,
                shuffle = False,
                device = 'cpu',
                criterion = torch.nn.functional.mse_loss,
                *args,
                **kwargs):

    if verbose_train: print('\n ------- Begin training process ------- \n')

    # Hardware
    if device is None: device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
    device = torch.device(device)
    torch.cuda.empty_cache()

    # Setups
    history = {'train_loss': [], 'val_loss': [], 'time_4_epoch': [], 'step_size': [],
               'accepted': [], 'nfev': 0, 'Exit':[]}
    dataset = create_dataset(ds)
    input_dim = dataset.n
    layers = set_architecture(arch,input_dim,seed)
    model = FNN(layers=layers).to(device)
    opt = 'cmafd'
    optimizer = set_optimizer(opt,model,*args,**kwargs)

    if verbose_train: print("\n --------- Start Train --------- \n")

    # Train
    start_time_4_epoch = time.time()
    fw0 = closure(dataset,device,model,criterion)
    optimizer.set_fw0(fw0)
    history['nfev'] +=1
    f_before = fw0
    optimizer.set_reference(f_before=f_before)

    test_loss = closure(dataset,device,model,criterion,test=True)
    history['train_loss'].append(f_before)
    history['val_loss'].append(test_loss)
    history['time_4_epoch'].append(0.0)
    history['step_size'].append(optimizer.param_groups[0]['zeta'])

    # Training cycle - Epochs Level
    for epoch in range(ep):
        if verbose_train: print(f'Epoch n. {epoch+1}/{ep}')
        if time.time() - start_time_4_epoch >= time_limit:
            break

        w_before = get_w(model)

        # Training cycle - Single epoch level
        n_iterations = int(np.ceil(dataset.P/batch_size))
        f_tilde = 0
        for j in range(n_iterations):
            optimizer.zero_grad()
            #print(f'j= {j}, n_iterations= {n_iterations}, P ={dataset.P}, batch_size= {batch_size} ')
            dataset.minibatch(j * batch_size, (j + 1) * batch_size)
            x, y = dataset.x_train_mb.to(device), dataset.y_train_mb.to(device)
            #print(f'shape_x= {x.shape}, shape_y= {y.shape}')
            y_pred = model(x)
            loss = criterion(target=y, input=y_pred)
            f_tilde += loss.item() * (len(x) / dataset.P)
            loss.backward()
            #if verbose_train: print(f'Batch {j+1}/{n_iterations}   Running Loss: {loss.item():.6f}')
            optimizer.step()



        model, history, f_before, f_after, exit_type = optimizer.control_step(model,w_before,closure,
                                                dataset,device,criterion,history,epoch)
        optimizer.set_reference(f_before=f_after)
        elapsed_time_4_epoch = time.time() - start_time_4_epoch

        if history['step_size'][-1] <= 1e-15:
            history['comments'] = f'Forced stopping at epoch {epoch}'
            break




        # Test
        test_loss = closure(dataset,device,model,criterion,test = True)

        # Update history
        try:
            history['train_loss'].append(f_after.item())
        except:
            history['train_loss'].append(f_after)
        try:
            history['val_loss'].append(test_loss.item())
        except:
            history['val_loss'].append(test_loss)
        history['time_4_epoch'].append(elapsed_time_4_epoch)
        if verbose_train: print(f'End Epoch {epoch}   Train Loss:{f_after:3e}  Elapsed time:{elapsed_time_4_epoch:3f} \n ')


        # Empty CUDA cache
        torch.cuda.empty_cache()

        #If needed, reshuffle
        if shuffle == True:
            dataset.reshuffle(seed=random.randint(1,1000))

    # Operations after training
    torch.save(history,sm_root + 'history_'+opt+'_'+arch+'_'+ds+'_'+ID_history+'.txt')
    if verbose_train: print('\n - Finished Training - \n')
    return history


if __name__ == "__main__":
    dataset_list = [ds[:-4] for ds in os.listdir('dataset')]
    #algorithms = ['lbfgs','ig','cma','nmcma']
    architectures = ['S','XXL','XXXL','4XL','M','L','XL']
    seeds = [1]#,10,100,1000,10000]
    all_probs = [(ds,net) for ds in dataset_list for net in architectures]
    #all_probs = [('Mv','S')]
    smroot = 'prove_CMAL/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device == ',device)

    print('---------------- CMA-FD -----------------')
    for idx, problem in enumerate(all_probs):
     print(f'Solving Problem {idx + 1}/{len(all_probs)} --- Dataset: {problem[0]}   Network: {problem[1]}')
     for j, seed in enumerate(seeds):
         print(f'Run {j + 1}/{len(seeds)}')
         history = train_model(ds=problem[0],
                               arch=problem[1],
                               sm_root=smroot,
                               ep=1000000,
                               time_limit=100,
                               max_it_EDFL=50,
                               ID_history='seed_' + str(seed),
                               verbose=True,
                               batch_size=128,
                               verbose_EDFL=True,
                               verbose_train=True,
                               seed=seed,
                               device=device)







