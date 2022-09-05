# MagNet
MagNet: A Neural Network for Directed Graphs

```
@article{zhang2021magnet,
  title={Magnet: A neural network for directed graphs},
  author={Zhang, Xitong and He, Yixuan and Brugnone, Nathan and Perlmutter, Michael and Hirn, Matthew},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={27003--27015},
  year={2021}
}
```

## Environment Setup
### Overview
The project has been tested on the following environment specification:
1. Ubuntu 18.04.5 LTS (Other x86_64 based Linux distributions should also be fine, such as Fedora 32)
2. Nvidia Graphic Card (NVIDIA GeForce RTX 2080 with driver version 440.36, and NVIDIA RTX 8000) and CPU (Intel Core i7-10700 CPU @ 2.90GHz)
3. Python 3.6.13 (and Python 3.6.12)
4. CUDA 10.2 (and CUDA 9.2)
5. Pytorch 1.8.0 (built against CUDA 10.2) and Python 1.6.0 (built against CUDA 9.2)
6. Other libraries and python packages (See below)

> The implementation of MagNet without baselines: [SimpleMagNet](https://github.com/XitongZhang1994/SimpleMagNet)

### Installation
*There are two installation methods listed below. Please try to install the packages manually if the first method fails.*

#### Installation method 1 (.yml files)
You should handle (1),(2) yourself. For (3), (4), (5) and (6), we provide a list of steps to install them.

We provide two examples of envionmental setup, one with CUDA 10.2 and GPU, the other with CPU.

Following steps assume you've done with (1) and (2).
1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Both Miniconda and Anaconda are OK.

2. Create an environment and install python packages (GPU):
```
conda env create -f environment_GPU.yml
```

3. Create an environment and install python packages (CPU):
```
conda env create -f environment_CPU.yml
```


#### Installation method 2 (manual installation)
The codebase is implemented in Python 3.6.12. package versions used for development are below.
```
networkx           2.5
numpy              1.19.2
pandas             1.1.4
scipy              1.5.4
argparse           1.1.0
sklearn            0.23.2
stellargraph       1.2.1 (for link direction prediction: conda install -c stellargraph stellargraph)
torch              1.8.0
torch-scatter      2.0.5
torch-geometric    1.6.3 (follow https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
matplotlib         3.3.4 (for generating plots and results)
```

### Execution checks
When installation is done, you could check you enviroment via:
```
cd execution
bash setup_test.sh
```

## Folder structure
- ./execution/ stores files that can be executed to generate outputs. For vast number of experiments, we use parallel (https://www.gnu.org/software/parallel/, can be downloaded in command line and make it executable via:
```
wget http://git.savannah.gnu.org/cgit/parallel.git/plain/src/parallel
chmod 755 ./parallel
```

- ./joblog/ stores job logs from parallel. 
You might need to create it by 
```
mkdir joblog
```

- ./Output/ stores raw outputs (ignored by Git) from parallel.
You might need to create it by 
```
mkdir Output
```

- ./dataset/ stores processed data sets.

- ./src/ stores files to train various models, utils and metrics.

- ./result_arrays/ stores results for different data sets. Each data set has a separate subfolder.

- ./result_anlysis/ stores notebooks for generating result plots or tables.

- ./logs/ stores trained models and logs, as well as predicted clusters (optional). When you are in debug mode (see below), your logs will be stored in ./debug_logs/ folder.

## Options
<p align="justify">
MagNet provides the following command line arguments, which can be viewed in the ./src/param_parser.py.
</p>

### Synthetic data options for node classification task:
See file ./src/sparse_Magnet.py for example.
```
  --p_q                   FLOAT         Direction strength, from 0.5 to 1.                      Default is 0.95. 
  --p_inter               FLOAT         Inter-cluster edge probabilities.                       Default is 0.1.
  --seed                  INT           Random seed for training testing 
                                            split/random graph generation.                      Default is 0.
```

### Major model options for node classification:
See file ./src/sparse_Magnet.py for example.

```
  --epochs                INT         Number of (maximum) training epochs.                      Default is 3000. 
  --q                     FLOAT       q value for the phase matrix.                             Default is 0. 
  --new_setting, -NS      BOOL        Whether not to load best settings.                        Default is False.
  --num_filter            INT         Numer of filters.                                         Default is 1.
  --layer                 INT         Numer of layers.                                          Default is 2.
  --lr                    FLOAT       Learning rate.                                            Default is 0.005.  
  --l2                    FLOAT       Weight decay (L2 loss on parameters).                     Default is 5^-4. 
  --dropout               FLOAT       Dropout rate (1 - keep probability).                      Default is 0.0.
  --debug, -D             BOOL        Debug with minimal training setting, not to get results.  Default is False.
  --dataset               STR         Data set to consider.                                     Default is 'WebKB/Cornell'.
```

### Major model options for link prediction:
See file ./src/Edge_sparseMagnet.py for example.

```
  --epochs                INT         Number of (maximum) training epochs.                      Default is 1500. 
  --num_class_link        INT         Number of classes for link direction prediction(2 or 3).  Default is 2.
  --task                  INT         Task to conduct: 1 is existence prediction while 2 is
                                        link (direction) prediction, could be 2 or 3 classes.   Default is 2.
```

## Reproduce results
First, get into the ./execution/ folder:
```
cd execution
```
To reproduce synthetic results on node classification on cyclic meta-graph structure.
```
bash cyclic.sh
```
To reproduce results on link prediction.
```
bash link_prediction.sh
```
Other execution files are similar to run.

Note that if you are operating on CPU, you may delete the commands ``CUDA_VISIBLE_DEVICES=xx". You can also set you own number of parallel jobs, not necessarily following the j numbers in the .sh files.

## Direct execution with training files

First, get into the ./src/ folder:
```
cd src
```

Then, below are various options to try:

Creating a MagNet model of the default setting.
```
python ./sparse_Magnet.py
```
Creating an APPNP model for cyclic meta-graph synthetic data with seed 10 and do not load "best setting".
```
python ./APPNP.py --dataset syn/cyclic --seed 10 -NS
```
Creating a model for Telegram data set on DGCN with some custom learning rate and epoch number.
```
python ./Sym_DiGCN.py --dataset telegram --lr 0.001 --epochs 300
```
Creating a model on Digraph with inception block for link prediction on a binary classification problem of direction prediction.
```
python ./Digraph.py --method_name DiG_ib --task 2 --num_class_link 2
```
### Generate Synthetic Graphs
```
cd src/utils
```
To generate cyclic DSBM graphs:
```
python generate_cyclic.py
```
To generate ordered DSBM graphs:
```
python generate_complete.py
python generate_syn2_3.py
```
To generate noisy cyclic DSBM graphs:
```
python generate_fill.py
```
## Notes

- Notations. Sym_DiGCN corresponds to DGCN. Digraph corresponids to two variants, specified by method names "DiG" and "DiG_ib", respectively.

- When tuning hyperparameters on synthetic data for the node classification task, we need "-NS" to ignore the auto loading of saved dictionaries of best setting. When we are running multiple experiments on different generated networks under the same synthetic setting, we do not put "-NS" into the command but we only need to specify the data we use and the p_q values etc.. The .py files will automatically load the best setting for us. To change the best setting, you could create or update the dictionaries, or simply add "-NS" and specify your setting of choice.
--------------------------------------------------------------------------------
