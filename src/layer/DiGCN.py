#############################################
# Copy from DiGCN 
# https://github.com/flyingtango/DiGCN
#############################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, add_self_loops
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.nn.inits import glorot, zeros

class DIGCNConv(MessagePassing):
    r"""The graph convolutional operator takes from Pytorch Geometric.
    The spectral operation is the same with Kipf's GCN.
    DiGCN preprocesses the adjacency matrix and does not require a norm operation during the convolution operation.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the adj matrix on first execution, and will use the
            cached version for further executions.
            Please note that, all the normalized adj matrices (including undirected)
            are calculated in the dataset preprocessing to reduce time comsume.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, improved=False, cached=True,
                 bias=True, **kwargs):
        super(DIGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None
    
    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if edge_weight is None:
                raise RuntimeError(
                    'Normalized adj matrix cannot be None. Please '
                    'obtain the adj matrix in preprocessing.')
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class DiModel(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout = False, layer=2):
        super(DiModel, self).__init__()
        self.conv1 = DIGCNConv(input_dim, filter_num)
        self.conv2 = DIGCNConv(filter_num, filter_num)
        
        self.layer = layer
        if layer == 3:
            self.conv3 = DIGCNConv(filter_num, filter_num)

        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        if self.layer==3:
            x = F.relu(self.conv3(x, edge_index, edge_weight))

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class DiGCNet(torch.nn.Module):
    def __init__(self, input_dim, out_dim, hidden, dropout = False):
        super(DiGCNet, self).__init__()
        self.conv1 = DIGCNConv(input_dim, hidden)
        self.conv2 = DIGCNConv(hidden, hidden)
        self.linear = nn.Linear(hidden*2, out_dim)     
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        
        x = torch.cat((x[index[:,0]], x[index[:,1]]), axis=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)

class InceptionBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(InceptionBlock, self).__init__()
        self.ln = Linear(in_dim, out_dim)
        self.conv1 = DIGCNConv(in_dim, out_dim)
        self.conv2 = DIGCNConv(in_dim, out_dim)
    def reset_parameters(self):
        self.ln.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    def forward(self, x, edge_index, edge_weight, edge_index2, edge_weight2):
        x0 = self.ln(x)
        x1 = self.conv1(x, edge_index, edge_weight)
        x2 = self.conv2(x, edge_index2, edge_weight2)
        return x0, x1, x2

class DiGCN_IB(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer = 2):
        super(DiGCN_IB, self).__init__()
        self.ib1 = InceptionBlock(num_features, hidden)
        self.ib2 = InceptionBlock(hidden, hidden)
        self._dropout = dropout
        self.Conv = nn.Conv1d(hidden, num_classes, kernel_size=1)

        self.layer = layer
        if layer == 3:
            self.ib3 = InceptionBlock(hidden, hidden)

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0,x1,x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0+x1+x2
        x0,x1,x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0+x1+x2
        if self.layer == 3:
            x0,x1,x2 = self.ib3(x, edge_index, edge_weight, edge_index2, edge_weight2)
            x = x0+x1+x2

        x = F.dropout(x, p=self._dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()
        return F.log_softmax(x, dim=1)

class DiGCNet_IB(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden, dropout = False):
        super(DiGCNet_IB, self).__init__()
        self.ib1 = InceptionBlock(num_features, hidden)
        self.ib2 = InceptionBlock(hidden, hidden)
        self.linear = nn.Linear(hidden*2, num_classes)   
        self.dropout = dropout

    def forward(self, features, edge_index_tuple, index, edge_weight_tuple):
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0,x1,x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0+x1+x2
        x0,x1,x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0+x1+x2
        
        x = torch.cat((x[index[:,0]], x[index[:,1]]), axis=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)