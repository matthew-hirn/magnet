import torch
import numpy as np
import networkx as nx
from scipy import sparse
'''
def sym_distochastic(N = 1000,
                    prob = 0.5, 
                    q = 0.95, # Probability of directing the edge from A to B
                    directed = True,
                    todense = True,
                    seed = None):
    
    G = nx.generators.random_graphs.erdos_renyi_graph(N, prob, seed = seed)
    original_A = nx.adjacency_matrix(G).todense()
    
    new_A = np.zeros((N, N))
    for i in range(N):
        for j in range(0,i):
            if (i >= N/2) and (j < N/2):
                if np.random.uniform(0,1) > q:
                    new_A[i,j] = 2 * original_A[i,j]
                else:
                    new_A[j,i] = 2 * original_A[i,j]
            else:
                if np.random.uniform(0,1) > 0.5:
                    new_A[i,j] = 2 * original_A[i,j]
                else:
                    new_A[j,i] = 2 * original_A[i,j]
    label = np.r_[np.ones(int(N/2)), np.zeros(int(N/2))]
    return new_A, label.astype('int')
'''
def desymmetric_stochastic(sizes = [100, 100, 100],
                probs = [[0.5, 0.45, 0.45],
                         [0.45, 0.5, 0.45],
                         [0.45, 0.45, 0.5]],
                seed = 0,
                off_diag_prob = 0.9, 
                norm = False):
    from sklearn.model_selection import train_test_split
    
    g = nx.stochastic_block_model(sizes, probs, seed=seed)
    original_A = nx.adjacency_matrix(g).todense()
    A = original_A.copy()
    
    # for blocks represent adj within clusters
    accum_size = 0
    for s in sizes:
        x, y = np.where(np.triu(original_A[accum_size:s+accum_size,accum_size:s+accum_size]))
        x1, x2, y1, y2 = train_test_split(x, y, test_size=0.5)
        A[x1+accum_size, y1+accum_size] = 0
        A[y2+accum_size, x2+accum_size] = 0
        accum_size += s

    # for blocks represent adj out of clusters (cluster2cluster edges)
    accum_x, accum_y = 0, 0
    n_cluster = len(sizes)
    
    for i in range(n_cluster):
        accum_y = accum_x + sizes[i]
        for j in range(i+1, n_cluster):
            x, y = np.where(original_A[accum_x:sizes[i]+accum_x, accum_y:sizes[j]+accum_y])
            x1, x2, y1, y2 = train_test_split(x, y, test_size=off_diag_prob)
            
            A[x1+accum_x, y1+accum_y] = 0
            A[y2+accum_y, x2+accum_x] = 0
                
            accum_y += sizes[j]
            
        accum_x += sizes[i]
    # label assignment based on parameter sizes 
    label = []
    for i, s in enumerate(sizes):
        label.extend([i]*s)
    label = np.array(label)      

    return np.array(original_A), np.array(A), label

def to_dataset(A, label, save_path):
    import pickle as pk
    from numpy import linalg as LA
    from Citation import train_test_split
    from torch_geometric.data import Data

    masks = {}
    masks['train'], masks['val'], masks['test'] = [], [] , []
    for split in range(10):
        mask = train_test_split(label, seed=split, train_examples_per_class=10, val_size=500, test_size=None)
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
    s = np.random.normal(0, 1.0, (len(A), 1))
    features = torch.from_numpy(s).float()

    data = Data(x=features, edge_index=indices, edge_weight=None, y=label)
    data.train_mask = torch.cat(masks['train'], axis=-1) 
    data.val_mask   = torch.cat(masks['val'], axis=-1)
    data.test_mask  = torch.cat(masks['test'], axis=-1)

    pk.dump(data, open(save_path, 'wb'))
    return

def main():
    node = 500
    cluster = 5
    sizes = [node]*cluster

    p_in, p_inter = 0.1, 0.1
    prob = np.diag([p_in]*cluster)
    prob[prob == 0] = p_inter

    for p_q in [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6]:
        for seed in [0, 10, 20, 30, 40]:
            _, A, label = desymmetric_stochastic(sizes = sizes, probs = prob, off_diag_prob = p_q, seed=seed)
            to_dataset(A, label, save_path = '../../dataset/data/tmp/syn/syn'+str(int(100*p_q))+'Seed'+str(seed)+'.pk')

    return

if __name__ == "__main__":
    main()