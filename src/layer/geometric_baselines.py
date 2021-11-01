import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, GINConv, APPNP

####################################################################
# Link Prediction Models
####################################################################
'''
def pairwise_similar(x):
    x = torch.tanh(x)
    xx = torch.exp(torch.matmul(x, x.T))
    xx = xx - torch.diag(torch.diag(xx, 0))
    return xx, torch.sum(xx, 1)+1e-8
'''

class APPNP_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, alpha = 0.1, dropout = False, K=1):
        super(APPNP_Link, self).__init__()
        self.dropout = dropout

        self.line1 = nn.Linear(input_dim, filter_num)
        self.line2 = nn.Linear(filter_num, filter_num)

        self.conv1 = APPNP(K=K, alpha=alpha)
        self.conv2 = APPNP(K=K, alpha=alpha)

        self.linear = nn.Linear(filter_num*2, out_dim)     

    def forward(self, x, edge_index, index):
        x = self.line1(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.line2(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = torch.cat((x[index[:,0]], x[index[:,1]]), axis=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)

class GIN_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout = False):
        super(GIN_Link, self).__init__()
        self.dropout = dropout
        self.line1 = nn.Linear(input_dim, filter_num)
        self.line2 = nn.Linear(filter_num, filter_num)

        self.conv1 = GINConv(self.line1)
        self.conv2 = GINConv(self.line2)
        self.linear = nn.Linear(filter_num*2, out_dim)     

    def forward(self, x, edge_index, index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = torch.cat((x[index[:,0]], x[index[:,1]]), axis=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)

class GCN_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout = False):
        super(GCN_Link, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, filter_num)
        self.conv2 = GCNConv(filter_num, filter_num)
        self.linear = nn.Linear(filter_num*2, out_dim)     

    def forward(self, x, edge_index, index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = torch.cat((x[index[:,0]], x[index[:,1]]), axis=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)

class Cheb_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, K, dropout = False):
        super(Cheb_Link, self).__init__()
        self.dropout = dropout
        self.conv1 = ChebConv(input_dim, filter_num, K)
        self.conv2 = ChebConv(filter_num, filter_num, K)
        self.linear = nn.Linear(filter_num*2, out_dim)     

    def forward(self, x, edge_index, index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = torch.cat((x[index[:,0]], x[index[:,1]]), axis=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)

class SAGE_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout = False):
        super(SAGE_Link, self).__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(input_dim, filter_num)
        self.conv2 = SAGEConv(filter_num, filter_num)
        #self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)
        self.linear = nn.Linear(filter_num*2, out_dim)     

    def forward(self, x, edge_index, index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = torch.cat((x[index[:,0]], x[index[:,1]]), axis=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)

class GAT_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, heads, filter_num, dropout = False):
        super(GAT_Link, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(input_dim, filter_num, heads=heads)
        self.conv2 = GATConv(filter_num*heads, filter_num, heads=heads)
        self.linear = nn.Linear(filter_num*heads*2, out_dim)     

    def forward(self, x, edge_index, index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = torch.cat((x[index[:,0]], x[index[:,1]]), axis=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)

'''
####################################################################
# Link Prediction Models in old versions of the paper
####################################################################
class Sym_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout = False):
        super(Sym_Link, self).__init__()
        self.dropout = dropout
        self.conv11 = GCNConv(input_dim, filter_num)
        self.conv12 = GCNConv(input_dim, filter_num)
        self.conv13 = GCNConv(input_dim, filter_num)
        
        self.conv21 = GCNConv(filter_num*3, filter_num)
        self.conv22 = GCNConv(filter_num*3, filter_num)
        self.conv23 = GCNConv(filter_num*3, filter_num)

        self.Conv = nn.Conv1d(filter_num*3, out_dim, kernel_size=1)

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, positive, negative):
        x1 = self.conv11(x, edge_index)
        x2 = self.conv12(x, edge_in, in_w)
        x3 = self.conv13(x, edge_out, out_w)
        x = torch.cat((x1, x2, x3), axis = -1)
        x = F.relu(x)

        x1 = self.conv21(x, edge_index)
        x2 = self.conv21(x, edge_in, in_w)
        x3 = self.conv23(x, edge_out, out_w)
        x = torch.cat((x1, x2, x3), axis = -1)
        x = F.relu(x)

        pos = x[positive[:,0]] - x[positive[:,1]]
        neg = x[negative[:,0]] - x[negative[:,1]]
        x = torch.cat((pos, neg), axis = 0)
        
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class APPNP_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, alpha = 0.1, dropout = False, K=1):
        super(APPNP_Link, self).__init__()
        self.dropout = dropout

        self.line1 = nn.Linear(input_dim, filter_num)
        self.line2 = nn.Linear(filter_num, filter_num)

        self.conv1 = APPNP(K=K, alpha=alpha)
        self.conv2 = APPNP(K=K, alpha=alpha)

        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

    def forward(self, x, edge_index, positive, negative):
        x = self.line1(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.line2(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        pos = x[positive[:,0]] - x[positive[:,1]]
        neg = x[negative[:,0]] - x[negative[:,1]]
        x = torch.cat((pos, neg), axis = 0)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class GIN_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout = False):
        super(GIN_Link, self).__init__()
        self.dropout = dropout
        self.line1 = nn.Linear(input_dim, filter_num)
        self.line2 = nn.Linear(filter_num, filter_num)

        self.conv1 = GINConv(self.line1)
        self.conv2 = GINConv(self.line2)

        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

    def forward(self, x, edge_index, positive, negative):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        pos = x[positive[:,0]] - x[positive[:,1]]
        neg = x[negative[:,0]] - x[negative[:,1]]
        x = torch.cat((pos, neg), axis = 0)
        
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class GCN_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout = False):
        super(GCN_Link, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, filter_num)
        self.conv2 = GCNConv(filter_num, filter_num)
        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

    def forward(self, x, edge_index, positive, negative):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        pos = x[positive[:,0]] - x[positive[:,1]]
        neg = x[negative[:,0]] - x[negative[:,1]]
        x = torch.cat((pos, neg), axis = 0)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class Cheb_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, K, dropout = False):
        super(Cheb_Link, self).__init__()
        self.dropout = dropout
        self.conv1 = ChebConv(input_dim, filter_num, K)
        self.conv2 = ChebConv(filter_num, filter_num, K)
        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

    def forward(self, x, edge_index, positive, negative):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        pos = x[positive[:,0]] - x[positive[:,1]]
        neg = x[negative[:,0]] - x[negative[:,1]]
        x = torch.cat((pos, neg), axis = 0)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class SAGE_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout = False):
        super(SAGE_Link, self).__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(input_dim, filter_num)
        self.conv2 = SAGEConv(filter_num, filter_num)
        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

    def forward(self, x, edge_index, positive, negative):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        pos = x[positive[:,0]] - x[positive[:,1]]
        neg = x[negative[:,0]] - x[negative[:,1]]
        x = torch.cat((pos, neg), axis = 0)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class GAT_Link(torch.nn.Module):
    def __init__(self, input_dim, out_dim, heads, filter_num, dropout = False):
        super(GAT_Link, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(input_dim, filter_num, heads=heads)
        self.conv2 = GATConv(filter_num*heads, filter_num, heads=heads)
        self.Conv = nn.Conv1d(filter_num*heads, out_dim, kernel_size=1)

    def forward(self, x, edge_index, positive, negative):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        pos = x[positive[:,0]] - x[positive[:,1]]
        neg = x[negative[:,0]] - x[negative[:,1]]
        x = torch.cat((pos, neg), axis = 0)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)
'''
####################################################################
# Node Classification Models
####################################################################
class GATModel(torch.nn.Module):
    def __init__(self, input_dim, out_dim, heads, filter_num, dropout = False, layer=2):
        super(GATModel, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(input_dim, filter_num, heads=heads)
        self.conv2 = GATConv(filter_num*heads, filter_num, heads=heads)
        self.Conv = nn.Conv1d(filter_num*heads, out_dim, kernel_size=1)
        self.layer = layer
        if layer == 3:
            self.conv3 = GATConv(filter_num*heads, filter_num, heads=heads)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        if self.layer == 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class SAGEModel(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout = False, layer=2):
        super(SAGEModel, self).__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(input_dim, filter_num)
        self.conv2 = SAGEConv(filter_num, filter_num)
        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

        self.layer = layer
        if layer == 3:
            self.conv3 = SAGEConv(filter_num, filter_num)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        if self.layer == 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout = False, layer=2):
        super(GCNModel, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, filter_num)
        self.conv2 = GCNConv(filter_num, filter_num)
        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

        self.layer = layer
        if layer == 3:
            self.conv3 = GCNConv(filter_num, filter_num)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        if self.layer == 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class ChebModel(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, K, dropout = False, layer=2):
        super(ChebModel, self).__init__()
        self.dropout = dropout
        self.conv1 = ChebConv(input_dim, filter_num, K)
        self.conv2 = ChebConv(filter_num, filter_num, K)
        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

        self.layer = layer
        if layer == 3:
            self.conv3 = ChebConv(filter_num, filter_num, K)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        if self.layer == 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class APPNP_Model(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, alpha = 0.1, dropout = False, layer=3):
        super(APPNP_Model, self).__init__()
        self.dropout = dropout
        self.line1 = nn.Linear(input_dim, filter_num)
        self.line2 = nn.Linear(filter_num, filter_num)

        self.conv1 = APPNP(K=10, alpha=alpha)
        self.conv2 = APPNP(K=10, alpha=alpha)
        self.layer = layer
        if layer == 3:
            self.line3 = nn.Linear(filter_num, filter_num)
            self.conv3 = APPNP(K=10, alpha=alpha)

        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.line1(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.line2(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        if self.layer == 3:
            x = self.line3(x)
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)

class GIN_Model(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout = False, layer=2):
        super(GIN_Model, self).__init__()
        self.dropout = dropout
        self.line1 = nn.Linear(input_dim, filter_num)
        self.line2 = nn.Linear(filter_num, filter_num)

        self.conv1 = GINConv(self.line1)
        self.conv2 = GINConv(self.line2)

        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)
        self.layer=layer
        if layer == 3:
            self.line3 = nn.Linear(filter_num, filter_num)
            self.conv3 = GINConv(self.line3)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        if self.layer == 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)
