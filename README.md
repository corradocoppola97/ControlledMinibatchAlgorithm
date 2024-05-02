__Controlled Minibatch Algorithm (CMA)__

This repository contains all the files needed to run and reproduce the results presented in the pre-print https://arxiv.org/abs/2212.01848 which is currently under peer-revision.

The results on __regression tasks__ can be reproduced after downloading the datasets from the drive below.
https://drive.google.com/drive/folders/1-MIe3ub6NaBRBiOIpX3cqXKjFhHuM00Q?usp=sharing

__Organization of the repository__

1) Dataset.py ---> Contains the wrapper to read and download the datasets. Please, ensure they are in the same repository or venv you are working from.
2) Model.py ---> Contains the different neural architectures.
3) [algo_name].py ---> This type of file contains the optimizer, respectively CMA, NMCMA, IG and L-BFGS. Note that L-BFGS has been taken from the official Pytorch library https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html and a time_limit condition has been added manually to terminate the optimization process when time_limit is exceeded.
4) PerformanceProfiles.py ---> To plot the performance and profiles from Dolan and More https://arxiv.org/abs/cs/0102001
5) main.py ---> Run regression tests on all the 11 dataset. Remind that the /Dataset folder must be in the same directory, from which you are running this file.
6) stats.py ---> To plot the main stats presented in the computational result sections (i.e., LS fail rate, IC acceptance rate, average nfev per epoch, average final step-size)
7) utility.py ---> Support functions.
