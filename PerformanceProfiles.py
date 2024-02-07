import os
import torch
import matplotlib.pyplot as plt
dataset_list = [ds[:-4] for ds in os.listdir('dataset')]
architectures = ['S','M','L','XL','XXL','XXXL']
all_probs = [(ds,net) for ds in dataset_list for net in architectures]
os.chdir('prove_reg')
algos = ['ig','cma','nmcma','lbfgs']


def find_fL(problem,seed):
    ds, net = problem
    bests = {}
    for algo in algos:
        file = 'history_'+algo+'_'+net+'_'+ds+'_seed_'+str(seed)+'.txt'
        stats = torch.load(file,map_location=torch.device('cpu'))
        bests[algo] = stats['train_loss'][-1]

    fL = min(bests.values())
    return fL, bests


def fw0(problem,seed):
    ds, net = problem
    algo = 'cma'
    file = 'history_' + algo + '_' + net + '_' + ds + '_seed_' + str(seed) + '.txt'
    stats = torch.load(file,map_location=torch.device('cpu'))
    fw0 = stats['train_loss'][0]
    return fw0


def c_sp(algo,problem,tau,seed):
    ds, net = problem
    initial_loss = fw0(problem, seed)
    fL = find_fL(problem,seed)[0]
    file = 'history_' + algo + '_' + net + '_' + ds + '_seed_' + str(seed) + '.txt'
    stats = torch.load(file,map_location=torch.device('cpu'))
    train_loss = stats['train_loss']
    elapsed_time = stats['time_4_epoch']
    csp = 3000
    for idx,loss in enumerate(train_loss):
        if loss <= fL + tau*(initial_loss-fL):
            csp = elapsed_time[idx]
    return csp

def r_sp(problem,tau,seed,solver):
    list_c = []
    c_ref = -1
    for algo in algos:
        csp = c_sp(algo,problem,tau,seed)
        if algo == solver:
            c_ref = csp
        list_c.append(csp)

    return c_ref/min(list_c)

def rho(solver,alpha,tau,seed,big_flag=False):
    rh = 0
    if big_flag == True:
        probs = [p for p in all_probs if p[0] in ['BlogFeedback','Covtype','Protein','Skin NonSkin','YearPredictionMSD']
                 and p[1] in ['XXXL','4XL','XXL']]
    else:
        probs = all_probs
    for problem in probs:
        rsp = r_sp(problem,tau,seed,solver)
        if rsp <= alpha:
            rh +=1
    return rh/len(probs)



plt.figure()
plt.xscale('log')
alphas = [1,2,4,8,16,32,64]
seeds = [1]#,10,100,1000,10000]
tau = 1e-1
colors = ['red','blue','green','orange']
mk = ['.','o','v','*']
idx = 0
for solver in algos:
    pp = []
    for alpha in range(1,65):
        r = sum(rho(solver,alpha,tau,seed,True) for seed in seeds)/len(seeds)
        pp.append(r)
    plt.plot([_ for _ in range(1,65)],pp,color=colors[idx],linestyle='solid',marker=mk[idx],markersize=7)
    idx +=1

plt.xticks(ticks=alphas,labels=[str(_) for _ in alphas])
plt.xlabel(chr(945))
plt.ylabel(chr(961) + '(' + chr(945) + ')')
plt.legend(algos)
plt.title(chr(964)+'='+str(tau)+' ')
plt.show()





