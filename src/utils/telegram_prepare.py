import math
import os
import random

import torch
import numpy as np
import networkx as nx
from scipy import sparse


def to_dataset(A, label, save_path, train_ratio=0.6, test_ratio=0.2):
    import pickle as pk
    from numpy import linalg as LA
    from Citation import train_test_split
    from torch_geometric.data import Data

    labels = label
    N = A.shape[0]
    idx = np.arange(N)
    num_clusters =  int(np.max(labels) + 1)
    clusters_sizes = [int(sum(labels==i)) for i in range(num_clusters)]
    test_sizes = [math.ceil(clusters_sizes[i] * test_ratio) for i in range(num_clusters)]
    val_ratio = 1 - train_ratio - test_ratio
    val_sizes = [math.ceil(clusters_sizes[i] * val_ratio) for i in range(num_clusters)]
    
    masks = {}
    masks['train'], masks['val'], masks['test'] = [], [] , []
    for _ in range(10):
        idx_test = []
        idx_val = []
        for i in range(num_clusters):
            idx_test_ind = random.sample(range(clusters_sizes[i]), k=test_sizes[i])
            idx_test.extend((np.array(idx)[labels==i])[idx_test_ind])
        idx_remain = list(set(idx).difference(set(idx_test))) # the rest of the indices
        clusters_sizes_remain = [int(sum(labels[idx_remain]==i)) for i in range(num_clusters)]
        for i in range(num_clusters):
            idx_val_ind = random.sample(range(clusters_sizes_remain[i]), k=val_sizes[i])
            idx_val.extend((np.array(idx_remain)[labels[idx_remain]==i])[idx_val_ind])
        idx_train = list(set(idx_remain).difference(set(idx_val))) # the rest of the indices

        train_indices = idx_train
        val_indices = idx_val
        test_indices = idx_test
        train_mask = np.zeros((labels.shape[0], 1), dtype=int)
        train_mask[train_indices, 0] = 1
        train_mask = np.squeeze(train_mask, 1)
        val_mask = np.zeros((labels.shape[0], 1), dtype=int)
        val_mask[val_indices, 0] = 1
        val_mask = np.squeeze(val_mask, 1)
        test_mask = np.zeros((labels.shape[0], 1), dtype=int)
        test_mask[test_indices, 0] = 1
        test_mask = np.squeeze(test_mask, 1)

        mask = {}
        mask['train'] = train_mask
        mask['val'] = val_mask
        mask['test'] = test_mask
        mask['train'] = torch.from_numpy(mask['train']).bool()
        mask['val'] = torch.from_numpy(mask['val']).bool()
        mask['test'] = torch.from_numpy(mask['test']).bool()
    
        masks['train'].append(mask['train'].unsqueeze(-1))
        masks['val'].append(mask['val'].unsqueeze(-1))
        masks['test'].append(mask['test'].unsqueeze(-1))
    
    label = torch.from_numpy(label).long()

    s_A = sparse.csr_matrix(A)
    coo = s_A.tocoo()
    values = coo.data
    
    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    '''
    A_sym = 0.5*(A + A.T)
    A_sym[A_sym > 0] = 1
    d_out = np.sum(np.array(A_sym), axis = 1)
    _, v = LA.eigh(d_out - A_sym)
    features = torch.from_numpy(np.sum(v, axis = 1, keepdims = True)).float()
    '''
    s = np.random.normal(0, 1.0, (s_A.shape[0], 1))
    features = torch.from_numpy(s).float()

    # data = Data(x=features, edge_index=indices, edge_weight=values, y=label)
    data = Data(x=features, edge_index=indices, edge_weight=None, y=label)
    data.train_mask = torch.cat(masks['train'], axis=-1) 
    data.val_mask   = torch.cat(masks['val'], axis=-1)
    data.test_mask  = torch.cat(masks['test'], axis=-1)

    pk.dump(data, open(save_path, 'wb'))
    return

def load_telegram():
    A = sparse.load_npz(os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../../dataset/data/tmp/telegram/telegram_adj.npz'))
    labels = np.load(os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../../dataset/data/tmp/telegram/telegram_labels.npy'))
    return A, labels

def main():  
    A, label = load_telegram()
    to_dataset(A, label, save_path = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../../dataset/data/tmp/telegram/telegram.pk'))

    return

if __name__ == "__main__":
    main()