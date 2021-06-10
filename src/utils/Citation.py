#############################################
# Copy and modify based on DiGCN 
# https://github.com/flyingtango/DiGCN
#############################################
import os.path as osp
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pandas as pd
import os
import torch
import sys
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected, to_networkx
from networkx.algorithms.components import is_weakly_connected

from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
from torch_scatter import scatter_add
import scipy
from torch_geometric.data import Dataset

def load_citation_link(root="./data"):
    g = load_npz_dataset(root)
    adj = g['A']
    coo = adj.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    
    data = Data(x=values, edge_index=indices, edge_weight=None, y=None)
    return [data]

def citation_datasets(root="./data", alpha=0.1, data_split = 10):
    # path = os.path.join(save_path, dataset)
    #os.makedirs(path, exist_ok=True)
    #dataset_path = os.path.join(path, '{}.npz'.format(dataset))
    g = load_npz_dataset(root)
    adj, features, labels = g['A'], g['X'], g['z']
    
    coo = adj.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    features = torch.from_numpy(features.todense()).float()

    # Set new random splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * the rest for testing
    masks = {}
    masks['train'], masks['val'], masks['test'] = [], [] , []
    for split in range(data_split):
        mask = train_test_split(labels, seed=split, train_examples_per_class=20, val_size=500, test_size=None)

        mask['train'] = torch.from_numpy(mask['train']).bool()
        mask['val'] = torch.from_numpy(mask['val']).bool()
        mask['test'] = torch.from_numpy(mask['test']).bool()
    
        masks['train'].append(mask['train'].unsqueeze(-1))
        masks['val'].append(mask['val'].unsqueeze(-1))
        masks['test'].append(mask['test'].unsqueeze(-1))

    labels = torch.from_numpy(labels).long()
    data = Data(x=features, edge_index=indices, edge_weight=None, y=labels)

    data.train_mask = torch.cat(masks['train'], axis=-1) 
    data.val_mask   = torch.cat(masks['val'], axis=-1)
    data.test_mask  = torch.cat(masks['test'], axis=-1)

    return [data]

def load_npz_dataset(file_name):
    """Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs

    """
    if not file_name.endswith('.npz'):
        file_name += file_name.split('/')[-2]+'.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        edge_index = loader['adj_indices'].copy()
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }

        idx_to_node = loader.get('idx_to_node')
        if idx_to_node:
            idx_to_node = idx_to_node.tolist()
            graph['idx_to_node'] = idx_to_node

        idx_to_attr = loader.get('idx_to_attr')
        if idx_to_attr:
            idx_to_attr = idx_to_attr.tolist()
            graph['idx_to_attr'] = idx_to_attr

        idx_to_class = loader.get('idx_to_class')
        if idx_to_class:
            idx_to_class = idx_to_class.tolist()
            graph['idx_to_class'] = idx_to_class

        return graph

def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples = labels.shape[0]
    num_classes = labels.max()+1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples = labels.shape[0]
    num_classes = labels.max()+1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices

def train_test_split(labels, seed, train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None, train_size=None, val_size=None, test_size=None):
    random_state = np.random.RandomState(seed)
    train_indices, val_indices, test_indices = get_train_val_test_split(
        random_state, labels, train_examples_per_class, val_examples_per_class, test_examples_per_class, train_size, val_size, test_size)

    #print('number of training: {}'.format(len(train_indices)))
    #print('number of validation: {}'.format(len(val_indices)))
    #print('number of testing: {}'.format(len(test_indices)))

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
    return mask

if __name__ == "__main__":
    data = citation_datasets(root="../../dataset/data/nips_data/cora_ml/raw/", dataset='cora_ml')
    print(data.train_mask.shape)
    # print_dataset_info()
    # get_npz_data(dataset='amazon_photo')
    ### already fixed split dataset!!!
    #if opt.dataset == 'all':
    #    for mode in ['cora', 'cora_ml','citeseer','dblp','pubmed']:
    #        get_npz_data(dataset = mode)
    #else:
    #    get_npz_data(dataset = opt.dataset)