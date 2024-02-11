import torch
from matplotlib import pyplot as plt
import os
import numpy as np
seeds = [10,100,1000,10000]
dataset_list = [ds[:-4] for ds in os.listdir('dataset')]
big_datasets = ['BlogFeedback', 'Covtype', 'Protein', 'Skin NonSkin', 'YearPredictionMSD']
small_datasets = [d for d in dataset_list if d not in big_datasets]
os.chdir('prove_reg')
architectures = ['S','M','L','XL','XXL','XXXL','4XL']
all_probs = [(ds,net) for ds in dataset_list for net in architectures]

def average_nfev(architecture, dataset_list, type):
    d = {}

    d['CMA'] = []
    for seed in seeds:
        for ds in dataset_list:
            file = 'history_cma_'+architecture+'_'+ds+'_seed_'+str(seed)+'.txt'
            stats = torch.load(file)
            avg_nfev = stats['nfev']/(len(stats['train_loss'])-1)
            d['CMA'].append(avg_nfev)

    d['NMCMA'] = []
    for seed in seeds:
        for ds in dataset_list:
            file = 'history_nmcma_'+architecture+'_'+ds+'_seed_'+str(seed)+'.txt'
            stats = torch.load(file)
            avg_nfev = stats['nfev']/(len(stats['train_loss'])-1)
            d['NMCMA'].append(avg_nfev)

    boxplot_result = plt.boxplot([d['CMA'],d['NMCMA']], showmeans=True, meanline=True, conf_intervals=np.array([[0.05, 0.95], [0.05, 0.95]]),
            boxprops=dict(color="blue"), medianprops=dict(color="black"), whiskerprops=dict(color="black"),
            capprops=dict(color="black"),showfliers=True)
    plt.xticks([1, 2], ['CMA', 'NMCMA'])
    plt.title(architecture + ' ' + type)
    #plt.xlabel('Keys')
    plt.ylabel('nfev')
    plt.ylim((1,2.75))
    boxplot_result['boxes'][1].set_color('red')
    plt.savefig('Avg_nfev_'+architecture+ '_' + type+'.pdf')
    plt.show()
    return d



def linesearch_failures(architecture, dataset_list, type):
    d = {}

    d['CMA'] = []
    for seed in seeds:
        for ds in dataset_list:
            file = 'history_cma_'+architecture+'_'+ds+'_seed_'+str(seed)+'.txt'
            stats = torch.load(file)
            ls_failures = len([_ for _ in stats['Exit'] if _ in {'13b','13c'}])
            d['CMA'].append(ls_failures)

    d['NMCMA'] = []
    for seed in seeds:
        for ds in dataset_list:
            file = 'history_nmcma_'+architecture+'_'+ds+'_seed_'+str(seed)+'.txt'
            stats = torch.load(file)
            ls_failures = len([_ for _ in stats['Exit'] if _ in {'16','10'}])
            d['CMA'].append(ls_failures)

    boxplot_result = plt.boxplot([d['CMA'],d['NMCMA']], showmeans=True, meanline=True, conf_intervals=np.array([[0.05, 0.95], [0.05, 0.95]]),
            boxprops=dict(color="blue"), medianprops=dict(color="black"), whiskerprops=dict(color="black"),
            capprops=dict(color="black"),showfliers=True)
    plt.xticks([1, 2], ['CMA', 'NMCMA'])
    plt.title(architecture + ' ' + type)
    #plt.xlabel('Keys')
    plt.ylabel('n_fail')
    plt.ylim((0,20))
    boxplot_result['boxes'][1].set_color('red')
    plt.savefig('Avg_LSfail_'+architecture+ '_' + type+'.pdf')
    plt.show()
    return d


def acceptance_inner_cycle(architecture, dataset_list, type):
    d = {}

    d['CMA'] = []
    for seed in seeds:
        for ds in dataset_list:
            file = 'history_cma_'+architecture+'_'+ds+'_seed_'+str(seed)+'.txt'
            stats = torch.load(file)
            avg_accepted = len(stats['accepted'])/(len(stats['train_loss'])-1)
            d['CMA'].append(avg_accepted)

    d['NMCMA'] = []
    for seed in seeds:
        for ds in dataset_list:
            file = 'history_nmcma_'+architecture+'_'+ds+'_seed_'+str(seed)+'.txt'
            stats = torch.load(file)
            avg_accepted = len(stats['accepted'])/(len(stats['train_loss'])-1)
            d['NMCMA'].append(avg_accepted)

    boxplot_result = plt.boxplot([d['CMA'],d['NMCMA']], showmeans=True, meanline=True, conf_intervals=np.array([[0.05, 0.95], [0.05, 0.95]]),
            boxprops=dict(color="blue",linewidth=7), medianprops=dict(color="black"), whiskerprops=dict(color="black"),
            capprops=dict(color="black"),showfliers=True)

    plt.xticks([1, 2], ['CMA', 'NMCMA'])
    plt.title(architecture + ' ' + type)
    #plt.xlabel('Keys')
    plt.ylabel('n_accepted')
    plt.ylim((0,1))
    boxplot_result['boxes'][1].set_color('red')
    plt.savefig('Avg_accepted_'+architecture+ '_' + type+'.pdf')
    plt.show()
    return d


def final_step_size(architecture, dataset_list, type):
    d = {}

    d['CMA'] = []
    for seed in seeds:
        for ds in dataset_list:
            file = 'history_cma_' + architecture + '_' + ds + '_seed_' + str(seed) + '.txt'
            stats = torch.load(file)
            zeta_final = stats['step_size'][-1]
            d['CMA'].append(zeta_final)

    d['NMCMA'] = []
    for seed in seeds:
        for ds in dataset_list:
            file = 'history_nmcma_' + architecture + '_' + ds + '_seed_' + str(seed) + '.txt'
            stats = torch.load(file)
            zeta_final = stats['step_size'][-1]
            d['NMCMA'].append(zeta_final)

    d['IG'] = []
    for seed in seeds:
        for ds in dataset_list:
            file = 'history_ig_' + architecture + '_' + ds + '_seed_' + str(seed) + '.txt'
            stats = torch.load(file)
            zeta_final = stats['step_size'][-1]
            d['IG'].append(zeta_final)

    boxplot_result = plt.boxplot([d['CMA'], d['NMCMA'], d['IG']], showmeans=True, meanline=True,
                                 conf_intervals=np.array([[0.05, 0.95], [0.05, 0.95],[0.05, 0.95]]),
                                 boxprops=dict(color="blue"), medianprops=dict(color="black"),
                                 whiskerprops=dict(color="black"),
                                 capprops=dict(color="black"), showfliers=True)

    plt.xticks([1, 2, 3], ['CMA', 'NMCMA', 'IG'])
    plt.title(architecture + ' ' + type)
    # plt.xlabel('Keys')
    plt.ylabel('nfev')
    #plt.ylim((1, 2.75))
    boxplot_result['boxes'][1].set_color('red')
    boxplot_result['boxes'][2].set_color('green')
    plt.savefig('Avg_zeta_' + architecture + '_' + type + '.pdf')
    plt.show()
    return d


if __name__ == '__main__':
    for arch in architectures:
        #average_nfev(arch,small_datasets,'small')
        #average_nfev(arch, big_datasets, 'big')
        #linesearch_failures(arch, small_datasets, 'small')
        #linesearch_failures(arch, big_datasets, 'big')
        acceptance_inner_cycle(arch, small_datasets, 'small')
        acceptance_inner_cycle(arch, big_datasets, 'big')
