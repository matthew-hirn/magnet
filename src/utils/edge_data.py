import torch, scipy
import numpy as np
import pickle as pk
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.utils import negative_sampling, dropout_adj, to_undirected

def sub_adj(edge_index, prob):
    sub_train, sub_test = train_test_split(edge_index.T, test_size = prob)
    sub_train, sub_val  = train_test_split(sub_train, test_size = 0.2)
    return sub_train.T, sub_val.T, sub_test.T

def edges_positive(edge_index):
    # return true edges and reverse edges
    return edge_index, edge_index[[1,0]]

def edges_negative(edge_index):
    size = edge_index.max().item() + 1
    adj = np.zeros((size, size), dtype=np.int8)
    adj[edge_index[0], edge_index[1]] = 1
    x, y = np.where((adj - adj.T) < 0)
    
    reverse = torch.from_numpy(np.c_[x[:,np.newaxis],y[:,np.newaxis]])
    undirected_index = to_undirected(edge_index)
    negative = negative_sampling(undirected_index, num_neg_samples=edge_index.shape[0], force_undirected=False)
    
    _from_, _to_ = negative[0].unsqueeze(0), negative[1].unsqueeze(0)
    neg_index = torch.cat((_from_, _to_), axis = 0)
    neg_index = torch.cat((reverse.T, neg_index), axis = 1)

    #print(edge_index.shape, reverse.shape, neg_index.shape)
    return neg_index

def split_negative(edge_index, prob):
    neg_index = edges_negative(edge_index)
    sub_train, sub_test = train_test_split(neg_index.T, test_size = prob)
    sub_train, sub_val  = train_test_split(sub_train, test_size = 0.2)
    return sub_train.T, sub_val.T, sub_test.T


def in_out_degree(edge_index, size):
    out_degree = torch.zeros((size, ))
    out_degree.scatter_add_(0, edge_index[0], torch.ones((edge_index.size(-1), )))

    in_degree = torch.zeros((size, ))
    in_degree.scatter_add_(0, edge_index[1], torch.ones((edge_index.size(-1), )))
     
    degree = torch.cat((in_degree.unsqueeze(0), out_degree.unsqueeze(0)), axis = 0).float()
    return degree.T

def generate_dataset(edge_index, save_path, splits = 10, test_prob = 0.6):
    datasets = {}
    
    for i in range(splits):
        train, val, test = sub_adj(edge_index, prob = test_prob)
        train_neg, val_neg, test_neg = split_negative(edge_index, prob = test_prob)
        ############################################
        # training data        
        ############################################
        # positive edges, reverse edges, negative edges
        datasets[i] = {}
        datasets[i]['train'] = {}
        datasets[i]['train']['positive'] = train
        datasets[i]['train']['negative'] = train_neg
        
        ############################################
        # validation data    
        ############################################
        # positive edges, reverse edges, negative edges
        datasets[i]['validate'] = {}
        datasets[i]['validate']['positive'] = val
        datasets[i]['validate']['negative'] = val_neg

        ############################################
        # test data    
        ############################################
        # positive edges, reverse edges, negative edges
        datasets[i]['test'] = {}
        datasets[i]['test']['positive'] = test
        datasets[i]['test']['negative'] = test_neg

    pk.dump(datasets, open(save_path, 'wb'), protocol=pk.HIGHEST_PROTOCOL)
    return datasets

#################################################################################
# Copy from DiGCN
# https://github.com/flyingtango/DiGCN
#################################################################################
def get_appr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight=None):
    from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
    from torch_scatter import scatter_add

    if edge_weight ==None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(edge_index.long(), edge_weight, fill_value, num_nodes)  
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes) 
    deg_inv = deg.pow(-1) 
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight 

    # personalized pagerank p
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
    p_v = torch.zeros(torch.Size([num_nodes+1,num_nodes+1]))
    p_v[0:num_nodes,0:num_nodes] = (1-alpha) * p_dense
    p_v[num_nodes,0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes,num_nodes] = alpha
    p_v[num_nodes,num_nodes] = 0.0
    p_ppr = p_v 

    eig_value, left_vector = scipy.linalg.eig(p_ppr.numpy(),left=True,right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    val, ind = eig_value.sort(descending=True)

    pi = left_vector[:,ind[0]] # choose the largest eig vector
    pi = pi[0:num_nodes]
    p_ppr = p_dense
    pi = pi/pi.sum()  # norm pi

    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi<0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag()

    # L_appr
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0

    # make nan to 0
    L[torch.isnan(L)] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L,as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]