__Controlled Minibatch Algorithm (CMA) and CMA Light__

This repository contains all the files needed to run and reproduce the results presented in the pre-print https://arxiv.org/abs/2212.01848 which is currently under peer-revision.

The results on __regression tasks__ can be reproduced after downloading the datasets from the drive below.
https://drive.google.com/drive/folders/1-MIe3ub6NaBRBiOIpX3cqXKjFhHuM00Q?usp=sharing

__Organization of the repository__

1) Dataset.py ---> Contains the wrapper to read and download the datasets. Please, ensure they are in the same repository or venv you are working from.
2) Model.py ---> Contains the different neural architectures.
3) Optimizers.py ---> Contains the optimizers CMA, IG, L-BFGS.
4) PerformanceProfiles.py ---> To plot the performance and profiles from Dolan and More https://arxiv.org/abs/cs/0102001
5) main.py ---> Run regression tests.
6) stats.py ---> To plot the main stats presented in the computational result sections (i.e., LS fail rate, IC acceptance rate, average nfev per epoch, average final step-size)
7) utility.py ---> Support functions.
