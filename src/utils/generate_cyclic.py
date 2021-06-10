import torch
import numpy as np
import networkx as nx
from scipy import sparse

'''
def desymmetric_stochastic_by_weight(sizes = [100, 100, 100],
                probs = [[0.5, 0.45, 0.45],
                         [0.45, 0.5, 0.45],
                         [0.45, 0.45, 0.5]],
                seed = 0,
                off_diag_prob = 0.9, 
                norm = False,
                cycle = False):
    from sklearn.model_selection import train_test_split
    g = nx.stochastic_block_model(sizes, probs, seed=seed)
    original_A = nx.adjacency_matrix(g).todense()
    A = original_A.copy().astype('float')
    
    # for blocks represent adj within clusters
    accum_size = 0
    for s in sizes:
        x, y = np.where(np.triu(original_A[accum_size:s+accum_size,accum_size:s+accum_size]))
        x1, x2, y1, y2 = train_test_split(x, y, test_size=0.5)
        A[x1+accum_size, y1+accum_size] = 0
        A[y2+accum_size, x2+accum_size] = 0
        
        A[accum_size:s+accum_size,accum_size:s+accum_size] = 0.5*A[accum_size:s+accum_size,accum_size:s+accum_size]
        
        accum_size += s

    # for blocks represent adj out of clusters (cluster2cluster edges)
    accum_x, accum_y = 0, 0
    n_cluster = len(sizes)
    
    for i in range(n_cluster):
        accum_y = accum_x + sizes[i]
        
        for j in range(i+1, n_cluster):
            if cycle and j>i+1: # for cycle clusters
                A[accum_x:sizes[i]+accum_x, accum_y:sizes[j]+accum_y] = 0
                A[accum_y:sizes[j]+accum_y, accum_x:sizes[i]+accum_x] = 0
                
            x, y = np.where(original_A[accum_x:sizes[i]+accum_x, accum_y:sizes[j]+accum_y])
            x1, x2, y1, y2 = train_test_split(x, y, test_size=0.5)
            
            A[x1+accum_x, y1+accum_y] = 0
            A[y2+accum_y, x2+accum_x] = 0
               
            A[accum_x:sizes[i]+accum_x, accum_y:sizes[j]+accum_y] = off_diag_prob*A[accum_x:sizes[i]+accum_x, accum_y:sizes[j]+accum_y]
            A[accum_y:sizes[j]+accum_y, accum_x:sizes[i]+accum_x] = (1-off_diag_prob)*A[accum_y:sizes[j]+accum_y, accum_x:sizes[i]+accum_x]
            
            accum_y += sizes[j]
        accum_x += sizes[i]
    
    # for cycle clusters
    accum_x = np.sum(sizes[:-1])
    if cycle:
        x, y = np.where(original_A[-sizes[-1]:, :sizes[0]])
        x1, x2, y1, y2 = train_test_split(x, y, test_size=0.5)
        
        tmp = original_A[-sizes[-1]:, :sizes[0]].copy()
        tmp[x1, y1] = 0
        A[-sizes[-1]:, :sizes[0]] = off_diag_prob*tmp.copy()
        
        tmp = original_A[:sizes[0], -sizes[-1]:].copy()
        tmp[y2, x2] = 0
        A[:sizes[0], -sizes[-1]:] = (1-off_diag_prob)*tmp.copy()

    # label assignment based on parameter sizes 
    label = []
    for i, s in enumerate(sizes):
        label.extend([i]*s)
    label = np.array(label)      

    return np.array(original_A), np.array(A), label
'''

def desymmetric_stochastic(sizes = [100, 100, 100],
                probs = [[0.5, 0.45, 0.45],
                         [0.45, 0.5, 0.45],
                         [0.45, 0.45, 0.5]],
                seed = 0,
                off_diag_prob = 0.9, 
                norm = False,
                cycle = False):
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
            if cycle and j>i+1: # for cycle clusters
                A[accum_x:sizes[i]+accum_x, accum_y:sizes[j]+accum_y] = 0
                A[accum_y:sizes[j]+accum_y, accum_x:sizes[i]+accum_x] = 0
                
            x, y = np.where(original_A[accum_x:sizes[i]+accum_x, accum_y:sizes[j]+accum_y])
            x1, x2, y1, y2 = train_test_split(x, y, test_size=off_diag_prob)
            
            A[x1+accum_x, y1+accum_y] = 0
            A[y2+accum_y, x2+accum_x] = 0
                
            accum_y += sizes[j]

        accum_x += sizes[i]

    # for cycle clusters
    accum_x = np.sum(sizes[:-1])
    if cycle:
        x, y = np.where(original_A[-sizes[-1]:, :sizes[0]])
        x1, x2, y1, y2 = train_test_split(x, y, test_size=off_diag_prob)
        
        tmp = original_A[-sizes[-1]:, :sizes[0]].copy()
        tmp[x1, y1] = 0
        A[-sizes[-1]:, :sizes[0]] = tmp.copy()
        
        tmp = original_A[:sizes[0], -sizes[-1]:].copy()
        tmp[y2, x2] = 0
        A[:sizes[0], -sizes[-1]:] = tmp.copy()

    # label assignment based on parameter sizes 
    label = []
    for i, s in enumerate(sizes):
        label.extend([i]*s)
    label = np.array(label)      

    return np.array(original_A), np.array(A), label

def to_dataset(A, label, save_path, train_examples_per_class = 10, val_size=500):
    import pickle as pk
    #from numpy import linalg as LA
    from Citation import train_test_split
    from torch_geometric.data import Data

    masks = {}
    masks['train'], masks['val'], masks['test'] = [], [] , []
    for split in range(10):
        mask = train_test_split(label, seed=split, train_examples_per_class=train_examples_per_class, val_size=val_size, test_size=None)
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

    data = Data(x=features, edge_index=indices, edge_weight=values, y=label)
    data.train_mask = torch.cat(masks['train'], axis=-1) 
    data.val_mask   = torch.cat(masks['val'], axis=-1)
    data.test_mask  = torch.cat(masks['test'], axis=-1)

    pk.dump(data, open(save_path, 'wb'))
    return

def add_symmetric_blocks(G, num_clst, p, q, add_to, seed): 
    # add the other cluster to G
    # G: the desymmetric graph
    # num_clst: # of cluster in G
    # add_to: # add to which cluster
    # p: the link probability inside the addtion cluster
    # q: the link probability between the addtion cluster and the others
    from sklearn.model_selection import train_test_split
    nodes = int(G.shape[0]/num_clst)
    # generate ER graph and desymmetrize 
    g = nx.erdos_renyi_graph(nodes, p, seed=seed)
    original_A = nx.adjacency_matrix(g).todense()
    A = original_A.copy()
    
    # for blocks represent adj within clusters
    x, y = np.where(np.triu(original_A))
    x1, x2, y1, y2 = train_test_split(x, y, test_size=0.5)
    A[x1, y1] = 0
    A[y2, x2] = 0
    
    rng = np.random.default_rng(seed)
    # generate inter-cluster edges
    block_r = rng.uniform(low=0.0, high=1.0, size=(nodes, nodes))
    block_r = (block_r < p).astype('int')
    block = block_r.copy()
    x, y = np.where(block)
    x1, x2, y1, y2 = train_test_split(x, y, test_size=0.5)
    block[x1, y1] = 0
    block_in   = block.copy()
    block_T_in = (block_r - block).T
    
    # generate in-cluster block
    # generate 2 blocks
    block_r = rng.uniform(low=0.0, high=1.0, size=(G.shape[0], nodes))
    block_r = (block_r < q).astype('int')
    
    block = block_r.copy()
    x, y = np.where(block)
    x1, x2, y1, y2 = train_test_split(x, y, test_size=0.5)
    block[x1, y1] = 0
    block_T = (block_r - block).T
    
    block[add_to*nodes:(add_to+1)*nodes,:] = block_in
    block_T[:,add_to*nodes:(add_to+1)*nodes] = block_T_in
    
    G       = np.c_[G,   block]
    block_T = np.c_[block_T, A]
    return np.r_[G, block_T]

def main():
    '''
    ####################
    #synthetic datasets
    ####################
    # synthetic datasets group #1 #
    node = 500
    cluster = 5
    p_q = 0.95
    sizes = [node]*cluster
    
    p_in, p_inter = 0.1, 0.1
    prob = np.diag([p_in]*cluster)
    prob[prob == 0] = p_inter
    _, A, label = desymmetric_stochastic(sizes = sizes, probs = prob, off_diag_prob = p_q, seed=0)
    to_dataset(A, label, save_path = '../../dataset/data/tmp/syn/syn1.pk')
    
    p_in, p_inter = 0.1, 0.08
    prob = np.diag([p_in]*cluster)
    prob[prob == 0] = p_inter
    _, A, label = desymmetric_stochastic(sizes = sizes, probs = prob, off_diag_prob = p_q, seed=0)
    to_dataset(A, label, save_path = '../../dataset/data/tmp/syn/syn2.pk')
    
    p_in, p_inter = 0.1, 0.05
    prob = np.diag([p_in]*cluster)
    prob[prob == 0] = p_inter
    _, A, label = desymmetric_stochastic(sizes = sizes, probs = prob, off_diag_prob = p_q, seed=0)
    to_dataset(A, label, save_path = '../../dataset/data/tmp/syn/syn3.pk')
    
    p_in, p_inter = 0.1, 0.1
    for i, p_q in enumerate([0.90, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]):
        prob = np.diag([p_in]*cluster)

        prob[prob == 0] = p_inter
        
        _, A, label = desymmetric_stochastic(sizes = sizes, probs = prob, off_diag_prob = p_q, seed=0)
        to_dataset(A, label, save_path = '../../dataset/data/tmp/syn/syn'+str(i+4)+'.pk')
    '''
    # synthetic datasets group #2 #
    # Cycle structured graphs     #
    # A --> B --> C --> D --> A   #
    node = 500
    cluster = 5
    sizes = [node]*cluster
    p_in, p_inter = 0.1, 0.1
    prob = np.diag([p_in]*cluster)
    prob[prob == 0] = p_inter

    for p_q in [0.95,0.9,0.85,0.8,0.75,0.7,0.65]:
        for seed in [0, 10, 20, 30, 40]:
            _, A, label = desymmetric_stochastic(sizes = sizes, probs = prob, off_diag_prob = p_q, seed=seed, cycle=True)
            to_dataset(A, label, save_path = '../../dataset/data/tmp/syn/cyclic'+str(int(100*p_q))+'Seed'+str(seed)+'.pk', 
            train_examples_per_class = 50)

    return

if __name__ == "__main__":
    main()