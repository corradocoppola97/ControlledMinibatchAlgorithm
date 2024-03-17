import matplotlib.pyplot as plt
import os
import torch
os.chdir('prove_CMAL')
algos = ['cma','cmal','adam','adagrad','adadelta']
networks = ['resnet18']#,'resnet34']#'resnet50']#,'resnet101','resnet152']#,'mobilenetv2']
seed = 1698064
dataset = 'cifar10'
colors = ['b','r','g','orange','black','yellow']
labels = ['CMA','CMA Light','Adam','Adagrad','Adadelta']
flag_epoch = True

if __name__ == '__main__':
    for network in networks:
        plt.figure()
        title = f'Training Loss - {network} on {dataset}'
        x_label = 'Epochs' if flag_epoch == True else 'Elapsed time'
        for i,algo in enumerate(algos):
            file = 'history_'+algo+'_'+network+'_'+dataset+'_seed_'+str(seed)+'.txt'
            history = torch.load(file)
            if flag_epoch == True:
                y_axis = history['train_loss']
                x_axis = [_ for _ in range(len(y_axis))]
            else:
                y_axis = history['train_loss']
                x_axis = history['time_4_epoch']
            plt.plot(x_axis,y_axis,color=colors[i],linewidth=1.15)
        plt.xlabel(x_label)
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend(labels)
        plt.savefig('Loss_'+dataset+'_'+network+'_'+x_label+'.pdf')

        plt.figure()
        title = f'Test Accuracy - {network} on {dataset}'
        x_label = 'Epochs' if flag_epoch == True else 'Elapsed time'
        for i,algo in enumerate(algos):
            file = 'history_'+algo+'_'+network+'_'+dataset+'_seed_'+str(seed)+'.txt'
            history = torch.load(file)
            if flag_epoch == True:
                y_axis = history['val_accuracy']
                x_axis = [_ for _ in range(len(y_axis))]
            else:
                y_axis = history['val_accuracy']
                x_axis = history['time_4_epoch']
            plt.plot(x_axis,y_axis,color=colors[i],linewidth=1.15)
        plt.xlabel(x_label)
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.legend(labels)
        plt.savefig('Accuracy_'+dataset+'_'+network+'_'+x_label+'.pdf')

