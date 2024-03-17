import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os


class Dataset:
    """
    Class to define the dasat and its properties
    """

    def __init__(self,
                 csvFile: str,
                 scaling: bool = False,
                 perc_test: float = 0.25):
        """
        :param csvFile: path to the dataset to read
        :param scaling: if True/1 data are scaled
        :param perc_test: percentage of data used for testing
        """
        if csvFile!='cifar10' and csvFile!='cifar100':
            self.dataset = pd.read_csv(csvFile)
            self.csv = csvFile.split('/')[-1]
            self.dataset = self.dataset.sample(frac=1, random_state=0)  # così ho fatto anche reshuffling

            if scaling == 1:
                scaler = MinMaxScaler()
                self.dataset = scaler.fit_transform(self.dataset)

            self.dataset = torch.Tensor(self.dataset).float()

            # extract train and test
            self.x = self.dataset[:, :-1]
            self.y = self.dataset[:, -1]

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=perc_test, random_state=1994)
            self.P = self.x_train.shape[0]
            self.n = self.x_train.shape[1]
            self.idx = None

        elif csvFile == 'cifar10':
            import torchvision
            transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                        torchvision.transforms.RandomRotation(10),
                                                        torchvision.transforms.RandomAffine(0, shear=10,
                                                                                            scale=(0.8, 1.2)),
                                                        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                                           saturation=0.2),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (
                                                            0.5, 0.5, 0.5))])

            ds_train = torchvision.datasets.CIFAR10(download=True, root='./', train=True,transform=transform)
            ds_test = torchvision.datasets.CIFAR10(download=True, root='./', train=False,transform=transform)
            train_loader = torch.utils.data.DataLoader(ds_train,batch_size=50000,shuffle=False)
            test_loader = torch.utils.data.DataLoader(ds_test, batch_size=10000, shuffle=False)
            for x,y in train_loader:
                self.x_train = x
                self.y_train = y
            for x, y in test_loader:
                self.x_test = x
                self.y_test = y
            self.P = self.x_train.shape[0]
            self.P_test = self.x_test.shape[0]
            self.n = 10
            self.idx = None

        elif csvFile == 'cifar100':
            import torchvision
            transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                        torchvision.transforms.RandomRotation(10),
                                                        torchvision.transforms.RandomAffine(0, shear=10,
                                                                                            scale=(0.8, 1.2)),
                                                        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                                           saturation=0.2),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (
                                                            0.5, 0.5, 0.5))])

            ds_train = torchvision.datasets.CIFAR100(download=True, root='./', train=True,transform=transform)
            ds_test = torchvision.datasets.CIFAR100(download=True, root='./', train=False,transform=transform)
            self.x_train = torch.tensor(ds_train.data).float()
            #self.x_train = torch.reshape(self.x_train,(-1,3,32,32))
            self.y_train = torch.tensor(ds_train.targets).float()
            self.x_test = torch.tensor(ds_test.data).float()
            #self.x_test = torch.reshape(self.x_test, (-1, 3,32,32))
            self.y_test = torch.tensor(ds_test.targets).float()
            self.P = self.x_train.shape[0]
            self.P_test = self.x_test.shape[0]
            self.n = 100
            self.idx = None
        else:
            raise SystemError('Dataset not available')

    def minibatch(self, first: int, last: int, test = False):
        """Extract a minibatch of observations from the starting dataset"""
        if test == False:
            if last > self.P:
                last = self.P
            self.x_train_mb = self.x_train[first:last, :]
            self.y_train_mb = self.y_train[first:last].flatten()
        else:
            self.x_test_mb = self.x_test[first:last, :]
            self.y_test_mb = self.y_test[first:last].flatten()

    def __str__(self):
        description = 'Dataset: {}.\nNumber of samples: {}. Number of variables: {}.'.format(self.csv, self.x.shape[0], self.x.shape[1])
        description = description + '\nTraining size: {}. Test size: {}.'.format(self.x_train.shape[1], self.x_test.shape[1])
        return description

    def get_idx(self,seed=100):
        '''
        Define the order in which the sample will be considered within the minibatches
        '''
        np.random.seed(seed)
        self.idx = np.arange(self.x_train.shape[0])
        np.random.shuffle(self.idx)
        return self.idx

    def reshuffle(self,seed=100):
        idx = self.get_idx(seed=seed)
        self.x_train = self.x_train[idx]
        self.y_train = self.y_train[idx]


def create_dataset(dataset: str):
    if dataset not in {'cifar10', 'cifar100'}:
        path = os.getcwd() +'/dataset/' + dataset+'.csv'
        csv_file_path = path
    else:
        csv_file_path = dataset
    try:
        dataset = Dataset(csv_file_path, scaling=True)
    except FileNotFoundError as e:
        print("data were not detected correctly. Check your cwd and the location of your dataset folder")
        print(f"cwd: {os.getcwd()}")
        raise e

    return dataset
