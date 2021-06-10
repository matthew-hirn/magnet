###############################################
# Modified from pytorch Geometric GCN
###############################################

from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

# the same as GCN but remove trainable weights
class DGCNConv(MessagePassing):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, 
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(DGCNConv, self).__init__(**kwargs)

        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.reset_parameters()

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SymModel(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout = False, layer = 2):
        super(SymModel, self).__init__()
        self.dropout = dropout
        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(filter_num*3, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim,    filter_num,   bias=False)
        self.lin2 = torch.nn.Linear(filter_num*3, filter_num, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, filter_num))
        self.bias2 = nn.Parameter(torch.Tensor(1, filter_num))

        self.layer = layer
        if layer == 3:
            self.lin3 = torch.nn.Linear(filter_num*3, filter_num, bias=False)
            self.bias3 = nn.Parameter(torch.Tensor(1, filter_num))
            nn.init.zeros_(self.bias3)

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w):
        x = self.lin1(x)
        x1 = self.gconv(x, edge_index)
        x2 = self.gconv(x, edge_in, in_w)
        x3 = self.gconv(x, edge_out, out_w)
        
        x1 += self.bias1
        x2 += self.bias1
        x3 += self.bias1

        x = torch.cat((x1, x2, x3), axis = -1)
        x = F.relu(x)

        x = self.lin2(x)
        x1 = self.gconv(x, edge_index)
        x2 = self.gconv(x, edge_in, in_w)
        x3 = self.gconv(x, edge_out, out_w)

        x1 += self.bias2
        x2 += self.bias2
        x3 += self.bias2

        x = torch.cat((x1, x2, x3), axis = -1)
        x = F.relu(x)

        if self.layer == 3:
            x = self.lin3(x)
            x1 = self.gconv(x, edge_index)
            x2 = self.gconv(x, edge_in, in_w)
            x3 = self.gconv(x, edge_out, out_w)

            x1 += self.bias3
            x2 += self.bias3
            x3 += self.bias3

            x = torch.cat((x1, x2, x3), axis = -1)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class Sym_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout = False):
        super(Sym_Link, self).__init__()
        self.dropout = dropout
        self.gconv = DGCNConv()

        self.lin1 = torch.nn.Linear(input_dim,    filter_num,   bias=False)
        self.lin2 = torch.nn.Linear(filter_num*3, filter_num, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, filter_num))
        self.bias2 = nn.Parameter(torch.Tensor(1, filter_num))
        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.linear = nn.Linear(filter_num*6, out_dim)  

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, index):
        x = self.lin1(x)
        x1 = self.gconv(x, edge_index)
        x2 = self.gconv(x, edge_in, in_w)
        x3 = self.gconv(x, edge_out, out_w)
        
        x1 += self.bias1
        x2 += self.bias1
        x3 += self.bias1

        x = torch.cat((x1, x2, x3), axis = -1)
        x = F.relu(x)

        x = self.lin2(x)
        x1 = self.gconv(x, edge_index)
        x2 = self.gconv(x, edge_in, in_w)
        x3 = self.gconv(x, edge_out, out_w)

        x1 += self.bias2
        x2 += self.bias2
        x3 += self.bias2

        x = torch.cat((x1, x2, x3), axis = -1)
        x = F.relu(x)

        #x = x[index[:,0]] - x[index[:,1]]
        x = torch.cat((x[index[:,0]], x[index[:,1]]), axis = -1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)