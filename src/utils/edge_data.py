import torch
import numpy as np
import pickle as pk
import networkx as nx
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from torch import Tensor
from torch_sparse import SparseTensor, coalesce
from stellargraph.data import EdgeSplitter
from sklearn.model_selection import train_test_split
from torch_geometric.utils import negative_sampling, dropout_adj
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_networkx
from networkx.algorithms.components import is_weakly_connected
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
from torch_scatter import scatter_add
import scipy
import os
from joblib import Parallel, delayed

def sub_adj(edge_index, prob, seed):
    sub_train, sub_test = train_test_split(edge_index.T, test_size = prob, random_state=seed)
    sub_train, sub_val  = train_test_split(sub_train, test_size = 0.2, random_state=seed)
    return sub_train.T, sub_val.T, sub_test.T

def edges_positive(edge_index):
    # return true edges and reverse edges
    return edge_index, edge_index[[1,0]]

def edges_negative(edge_index):
    from torch_geometric.utils import to_undirected

    size = edge_index.max().item() + 1
    adj = np.zeros((size, size), dtype=np.int8)
    adj[edge_index[0], edge_index[1]] = 1
    x, y = np.where((adj - adj.T) < 0)

    reverse = torch.from_numpy(np.c_[x[:,np.newaxis],y[:,np.newaxis]])
    undirected_index = to_undirected(edge_index)
    negative = negative_sampling(undirected_index, num_neg_samples=edge_index[0].shape[0], force_undirected=False)

    _from_, _to_ = negative[0].unsqueeze(0), negative[1].unsqueeze(0)
    neg_index = torch.cat((_from_, _to_), axis = 0)
    #neg_index = torch.cat((reverse.T, neg_index), axis = 1)
    #print(edge_index.shape, reverse.shape, neg_index.shape)
    return reverse.T, neg_index

def split_negative(edge_index, prob, seed, neg_sampling = True):
    reverse, neg_index = edges_negative(edge_index)
    if neg_sampling:
        neg_index = torch.cat((reverse, neg_index), axis = 1)
    else:
        neg_index = reverse

    sub_train, sub_test = train_test_split(neg_index.T, test_size = prob, random_state=seed)
    sub_train, sub_val  = train_test_split(sub_train, test_size = 0.2, random_state=seed)
    return sub_train.T, sub_val.T, sub_test.T

def label_pairs_gen(pos, neg):
    pairs = torch.cat((pos, neg), axis=-1)
    label = np.r_[np.ones(len(pos[0])), np.zeros(len(neg[0]))]
    return pairs, label

def generate_dataset_2class(edge_index, splits = 10, test_prob = 0.6):
    # this function doesn't consider the connectivity during removing edges for validation/testing
    from torch_geometric.utils import to_undirected
    datasets = {}
    
    for i in range(splits):

        train, val, test = sub_adj(edge_index, prob = test_prob, seed = i*10)
        train_neg, val_neg, test_neg = split_negative(edge_index, seed = i*10, prob = test_prob)
        ############################################
        # training data        
        ############################################
        # positive edges, reverse edges, negative edges
        datasets[i] = {}

        datasets[i]['graph'] = train
        datasets[i]['undirected'] = to_undirected(train).numpy().T

        rng = np.random.default_rng(i)

        datasets[i]['train'] = {}
        pairs, label = label_pairs_gen(train, train_neg)
        perm  = rng.permutation(len(pairs[0]))
        datasets[i]['train']['pairs'] = pairs[:,perm].numpy().T
        datasets[i]['train']['label'] = label[perm]
        
        ############################################
        # validation data    
        ############################################
        # positive edges, reverse edges, negative edges

        datasets[i]['validate'] = {}
        pairs, label = label_pairs_gen(val, val_neg)
        perm  = rng.permutation(len(pairs[0]))
        datasets[i]['validate']['pairs'] = pairs[:,perm].numpy().T
        datasets[i]['validate']['label'] = label[perm]
        ############################################
        # test data    
        ############################################
        # positive edges, reverse edges, negative edges
        
        datasets[i]['test'] = {}
        pairs, label = label_pairs_gen(test, test_neg)
        perm  = rng.permutation(len(pairs[0]))
        datasets[i]['test']['pairs'] = pairs[:,perm].numpy().T
        datasets[i]['test']['label'] = label[perm]
    return datasets

# in-out degree calculation
def in_out_degree(edge_index, size):
    A = coo_matrix((np.ones(len(edge_index)), (edge_index[:,0], edge_index[:,1])), shape=(size, size), dtype=np.float32).tocsr()
    out_degree = np.sum(A, axis = 0).T
    in_degree = np.sum(A, axis = 1)
    degree = torch.from_numpy(np.c_[in_degree, out_degree]).float()
    return degree

def undirected_label2directed_label(adj, edge_pairs, task):
    labels = np.zeros(len(edge_pairs), dtype=np.int32)
    new_edge_pairs = edge_pairs.copy()
    counter = 0
    for i, e in enumerate(edge_pairs): # directed edges
        if adj[e[0], e[1]] + adj[e[1], e[0]]  > 0: # exists an edge
            if adj[e[0], e[1]] > 0:
                if adj[e[1], e[0]] == 0: # rule out undirected edges
                    if counter%2 == 0:
                        labels[i] = 0
                        new_edge_pairs[i] = [e[0], e[1]]
                        counter += 1
                    else:
                        labels[i] = 1
                        new_edge_pairs[i] = [e[1], e[0]]
                        counter += 1
                else:
                    new_edge_pairs[i] = [e[0], e[1]]
                    labels[i] = -1
            else: # the other direction, and not an undirected edge
                if counter%2 == 0:
                    labels[i] = 0
                    new_edge_pairs[i] = [e[1], e[0]]
                    counter += 1
                else:
                    labels[i] = 1
                    new_edge_pairs[i] = [e[0], e[1]]
                    counter += 1
        else: # negative edges
            labels[i] = 2
            new_edge_pairs[i] = [e[0], e[1]]

    if task != 2:
        # existence prediction
        labels[labels == 2] = 1
        neg = np.where(labels == 1)[0]
        rng = np.random.default_rng(1000)
        neg_half = rng.choice(neg, size=len(neg)-np.sum(labels==0), replace=False)
        labels[neg_half] = -1
    return new_edge_pairs[labels >= 0], labels[labels >= 0]

def generate_dataset_3class(edge_index, size, save_path, splits = 10, probs = [0.15, 0.05], task = 2, label_dim = 2):
    save_file = save_path + 'task' + str(task) + 'dim'+ str(label_dim) + 'prob' + str(int(probs[0]*100)) + '_' + str(int(probs[1]*100)) + '.pk'
    if os.path.exists(save_file):
        print('File exists!')
        d_results = pk.load(open(save_file, 'rb'))
        return d_results

    row, col = edge_index[0], edge_index[1]
    #adj = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32).tocsr()
    #A_dense = np.array(adj.todense())
    #edge_num = np.sum(A_dense)
    #print( "undirected rate:", 1.0*np.sum(A_dense * A_dense.T)/edge_num )

    A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32).tocsr()
    G = nx.from_scipy_sparse_matrix(A) # create an undirected graph based on the adjacency

    def iteration(ind):
        datasets = {}
        edge_splitter_test = EdgeSplitter(G)
        G_test, ids_test, _ = edge_splitter_test.train_test_split(p=float(probs[0]), method="global", keep_connected=True, seed = ind)
        ids_test, labels_test = undirected_label2directed_label(A, ids_test, task)

        edge_splitter_val = EdgeSplitter(G_test)
        G_val, ids_val, _ = edge_splitter_val.train_test_split(p=float(probs[1]), method="global", keep_connected=True, seed = ind)
        ids_val, labels_val = undirected_label2directed_label(A, ids_val, task)

        edge_splitter_train = EdgeSplitter(G_val)
        _, ids_train, _ = edge_splitter_train.train_test_split(p=0.99, method="global", keep_connected=False, seed = ind)
        ids_train, labels_train = undirected_label2directed_label(A, ids_train, task)

        # observation after removing edges for training/validation/testing
        edges = [e for e in G_val.edges]
        # convert back to directed graph
        oberved_edges    = np.zeros((len(edges),2), dtype=np.int32)
        undirected_edges = np.zeros((2*len(G.edges),2), dtype=np.int32)
        
        for i, e in enumerate(edges):
            if A[e[0], e[1]] > 0:
                oberved_edges[i,0] = int(e[0])
                oberved_edges[i,1] = int(e[1])
            if A[e[1], e[0]] > 0:
                oberved_edges[i,0] = int(e[1])
                oberved_edges[i,1] = int(e[0])
        
        for i, e in enumerate(G.edges):
            if A[e[0], e[1]] > 0 or A[e[1], e[0]] > 0: 
                undirected_edges[i, :]            = [int(e[1]), e[0]]
                undirected_edges[i+len(edges), :] = [int(e[0]), e[1]]
        if label_dim == 2:
            ids_train = ids_train[labels_train < 2]
            labels_train = labels_train[labels_train <2]
            ids_test = ids_test[labels_test < 2]
            labels_test = labels_test[labels_test <2]
            ids_val = ids_val[labels_val < 2]
            labels_val = labels_val[labels_val <2]
        ############################################
        # training data        
        ############################################
        datasets[ind] = {}
        datasets[ind]['graph'] = torch.from_numpy(oberved_edges.T).long()
        datasets[ind]['undirected'] = undirected_edges
        
        datasets[ind]['train'] = {}
        datasets[ind]['train']['pairs'] = ids_train
        datasets[ind]['train']['label'] = labels_train
        ############################################
        # validation data    
        ############################################
        datasets[ind]['validate'] = {}
        datasets[ind]['validate']['pairs'] = ids_val
        datasets[ind]['validate']['label'] = labels_val
        ############################################
        # test data    
        ############################################
        datasets[ind]['test'] = {}
        datasets[ind]['test']['pairs'] = ids_test
        datasets[ind]['test']['label'] = labels_test
        return datasets
    
    # use larger n_jobs if the number of cpus is enough
    try:
        p_data = Parallel(n_jobs=4)(delayed(iteration)(ind) for ind in range(10))
    except:
        p_data = Parallel(n_jobs=1)(delayed(iteration)(ind) for ind in range(10))

    d_results = {}
    for ind in p_data:
        split = list(ind.keys())[0]
        d_results[split] = ind[split]
    
    if os.path.isdir(save_path) == False:
        try:
            os.makedirs(save_path)
        except FileExistsError:
            print('Folder exists!')

    if os.path.exists(save_file) == False:
        try:
            pk.dump(d_results, open(save_file, 'wb'), protocol=pk.HIGHEST_PROTOCOL)
        except FileExistsError:
            print('File exists!')

    return d_results

#################################################################################
# Copy from DiGCN
# https://github.com/flyingtango/DiGCN
#################################################################################
def get_appr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight=None):
    from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
    from torch_scatter import scatter_add

    if edge_weight is None:
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

def get_second_directed_adj(edge_index, num_nodes, dtype, edge_weight=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight 
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
    
    L_in = torch.mm(p_dense.t(), p_dense)
    L_out = torch.mm(p_dense, p_dense.t())
    
    L_in_hat = L_in
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0
    L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
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


@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    # type: (Tensor, Optional[int]) -> int
    pass


@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    # type: (SparseTensor, Optional[int]) -> int
    pass


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))

def to_undirected(edge_index, edge_weight=None, num_nodes=None):
    """Converts the graph given by :attr:`edge_index` to an undirected graph,
    so that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (FloatTensor, optional): The edge weights.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    if edge_weight is not None:
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
    edge_index, edge_weight = coalesce(edge_index, edge_weight,  num_nodes, num_nodes)

    return edge_index, edge_weight 

def link_prediction_evaluation(out_val, out_test, y_val, y_test):
    r"""Evaluates link prediction results.

    Args:
        out_val: (torch.FloatTensor) Log probabilities of validation edge output, with 2 or 3 columns.
        out_test: (torch.FloatTensor) Log probabilities of test edge output, with 2 or 3 columns.
        y_val: (torch.LongTensor) Validation edge labels (with 2 or 3 possible values).
        y_test: (torch.LongTensor) Test edge labels (with 2 or 3 possible values).

    :rtype: 
        result_array: (np.array) Array of evaluation results, with shape (2, 5).
    """
    out = torch.exp(out_val).detach().to('cpu').numpy()
    y_val = y_val.detach().to('cpu').numpy()
    # possibly three-class evaluation
    pred_label = np.argmax(out, axis = 1)
    val_acc_full = accuracy_score(pred_label, y_val)
    # two-class evaluation
    out = out[y_val < 2, :2]
    y_val = y_val[y_val < 2]


    prob = out[:,0]/(out[:,0]+out[:,1])
    prob = np.nan_to_num(prob, nan=0.5, posinf=0)
    val_auc = roc_auc_score(y_val, prob)
    pred_label = np.argmax(out, axis = 1)
    val_acc = accuracy_score(pred_label, y_val)
    val_f1_macro = f1_score(pred_label, y_val, average='macro')
    val_f1_micro = f1_score(pred_label, y_val, average='micro')

    out = torch.exp(out_test).detach().to('cpu').numpy()
    y_test = y_test.detach().to('cpu').numpy()
    # possibly three-class evaluation
    pred_label = np.argmax(out, axis = 1)
    test_acc_full = accuracy_score(pred_label, y_test)
    # two-class evaluation
    out = out[y_test < 2, :2]
    y_test = y_test[y_test < 2]
    

    prob = out[:,0]/(out[:,0]+out[:,1])
    prob = np.nan_to_num(prob, nan=0.5, posinf=0)
    test_auc = roc_auc_score(y_test, prob)
    pred_label = np.argmax(out, axis = 1)
    test_acc = accuracy_score(pred_label, y_test)
    test_f1_macro = f1_score(pred_label, y_test, average='macro')
    test_f1_micro = f1_score(pred_label, y_test, average='micro')
    return [[val_acc_full, val_acc, val_auc, val_f1_micro, val_f1_macro],
            [test_acc_full, test_acc, test_auc, test_f1_micro, test_f1_macro]]