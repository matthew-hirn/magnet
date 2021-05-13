# MagNet 
MagNet: A MagNetic Neural Network for Directed Graphs

In this paper, we propose MagNet, a spectral GNN for directed graphs based on a complex Hermitian matrix known as the magnetic Laplacian. This matrix encodes undirected geometric structure in the magnitude of its entries and directional information in the phase of its entries. 

[[Paper]](https://arxiv.org/abs/2102.11391)     

## environment setup
CUDA version == 10.2
pytorch version == 1.6.0
```
conda env create -f environment.yml
```
<!---
## notebook
--- 
The folder includes the jupyter-notebook scripts for debugging and visualization.
```
notebook (python version: 3.7.5)
+-- graph_gen.ipynb: 
|        1. random graph generation and visualization 
|        2. calculate hermitian adjacency
|        3. calculate hermitian laplacian
|        4. visualization
+-- layer.ipynb:
|        1. test the calculation process by numpy and torch
|        2. implement the torch layer demo 
|        3. implement the torch model
+-- cora_preprocessing.ipynb
|           load cora and preprocess it
+-- test_layer.ipynb
|           test if the implementation of diGCN outputs the same results as numpy implementation
+-- asymmetric_distance.ipynb
|           calculate asymmetric distance given node features
+-- flow_network.ipynb
|           get maxmimum flow given node feature matrix, the source and sink
+-- plot_date.ipynb
|            plot results based on the discussions on logs of date.
```
-->
## src
--- 
The folder includes the all related functions
```
src
+-- utlis:
|   +-- Citation.py
|   |       functions for syn1/sy2/syn3/cora/citeseer node classification
|   |       Thanks authors of Digraph Inception Convolutional Networks (NeurIPS 2020) https://github.com/flyingtango/DiGCN
|   +-- model_print.py
|   |       print the model structure
|   +-- preprocess.py
|   |       data preprocessing
|   +-- hermitian.py
|   |       hermitian decompostion and complex chebyshev-polynomial laplacian generation
|   +-- save_settings.py
|   |       save args and results to folder named by time
|   +-- symmetric_distochastic.py
|   |       generate directed "symmetric" stochastic block graphs, that A.T + A = A
|   |       group information includes clearly in the directed graphs but not undirected ones
|   +-- edge_data.py
|   |       edge data generation for link prediction

+-- layer:
|   +-- cheb.py
|   |       MagNet in ChebNet form
|   +-- DiGCN.py
|   |       Digraph baseline 
|   |       Thanks authors of Digraph Inception Convolutional Networks (NeurIPS 2020) https://github.com/flyingtango/DiGCN
|   +-- geometric_baselines.py
|   |       baselines based on Pytorch Geometric 
|   |       Thanks authors of the package https://github.com/rusty1s/pytorch_geometric
```

## Run
---
Link prediction:
```
cd src
python link_prediction.py
```
Node Classification:
```
cd src
python node_classification.py
```
Synthetic datasets generation
```
cd src/utils
python symmetric_distochastic.py
```
