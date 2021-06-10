import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
#from torch.nn import MultiheadAttention

def process(mul_L_real, mul_L_imag, weight, X_real, X_imag):
    data = torch.spmm(mul_L_real, X_real)
    real = torch.matmul(data, weight) 
    data = -1.0*torch.spmm(mul_L_imag, X_imag)
    real += torch.matmul(data, weight) 
    
    data = torch.spmm(mul_L_imag, X_real)
    imag = torch.matmul(data, weight)
    data = torch.spmm(mul_L_real, X_imag)
    imag += torch.matmul(data, weight)
    return torch.stack([real, imag])

class ChebConv(nn.Module):
    """
    The MagNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    :param L_norm_real, L_norm_imag: normalized laplacian of real and imag
    """
    def __init__(self, in_c, out_c, K,  L_norm_real, L_norm_imag, bias=True):
        super(ChebConv, self).__init__()

        L_norm_real, L_norm_imag = L_norm_real, L_norm_imag

        # list of K sparsetensors, each is N by N
        self.mul_L_real = L_norm_real   # [K, N, N]
        self.mul_L_imag = L_norm_imag   # [K, N, N]

        self.weight = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))  # [K+1, 1, in_c, out_c]

        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, data):
        """
        :param inputs: the input data, real [B, N, C], img [B, N, C]
        :param L_norm_real, L_norm_imag: the laplace, [N, N], [N,N]
        """
        X_real, X_imag = data[0], data[1]

        real = 0.0
        imag = 0.0

        future = []
        for i in range(len(self.mul_L_real)): # [K, B, N, D]
            future.append(torch.jit.fork(process, 
                            self.mul_L_real[i], self.mul_L_imag[i], 
                            self.weight[i], X_real, X_imag))
        result = []
        for i in range(len(self.mul_L_real)):
            result.append(torch.jit.wait(future[i]))
        result = torch.sum(torch.stack(result), dim=0)

        real = result[0]
        imag = result[1]
        return real + self.bias, imag + self.bias

class complex_relu_layer(nn.Module):
    def __init__(self, ):
        super(complex_relu_layer, self).__init__()
    
    def complex_relu(self, real, img):
        mask = 1.0*(real >= 0)
        return mask*real, mask*img

    def forward(self, real, img=None):
        # for torch nn sequential usage
        # in this case, x_real is a tuple of (real, img)
        if img == None:
            img = real[1]
            real = real[0]

        real, img = self.complex_relu(real, img)
        return real, img

class ChebNet(nn.Module):
    def __init__(self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim=2, activation=False, layer=2, dropout=False):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super(ChebNet, self).__init__()

        chebs = [ChebConv(in_c=in_c, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag)]
        if activation:
            chebs.append(complex_relu_layer())

        for i in range(1, layer):
            chebs.append(ChebConv(in_c=num_filter, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag))
            if activation:
                chebs.append(complex_relu_layer())

        self.Chebs = torch.nn.Sequential(*chebs)

        last_dim = 2  
        self.Conv = nn.Conv1d(num_filter*last_dim, label_dim, kernel_size=1)        
        self.dropout = dropout

    def forward(self, real, imag):
        real, imag = self.Chebs((real, imag))
        x = torch.cat((real, imag), dim = -1)
        
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = F.log_softmax(x, dim=1)
        return x

class ChebNet_Edge(nn.Module):
    def __init__(self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim = 2, activation = False, layer = 2, dropout = False):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super(ChebNet_Edge, self).__init__()
        
        chebs = [ChebConv(in_c=in_c, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag)]
        if activation and (layer != 1):
            chebs.append(complex_relu_layer())

        for i in range(1, layer):
            chebs.append(ChebConv(in_c=num_filter, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag))
            if activation:
                chebs.append(complex_relu_layer())
        self.Chebs = torch.nn.Sequential(*chebs)
        
        last_dim = 2
        self.linear = nn.Linear(num_filter*last_dim*2, label_dim)     
        self.dropout = dropout

    def forward(self, real, imag, index):
        real, imag = self.Chebs((real, imag))
        x = torch.cat((real[index[:,0]], real[index[:,1]], imag[index[:,0]], imag[index[:,1]]), dim = -1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x