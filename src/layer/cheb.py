import torch, math
import torch.nn as nn
import torch.nn.functional as F

class ChebConv(nn.Module):
    """
    The ChebNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    :param L_norm_real, L_norm_imag: normalized laplacian of real and imag
    """
    def __init__(self, in_c, out_c, K,  L_norm_real, L_norm_imag, bias=True):
        super(ChebConv, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))  # [K+1, 1, in_c, out_c]
        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        L_norm_real, L_norm_imag = L_norm_real, L_norm_imag
        self.mul_L_real = L_norm_real.unsqueeze(1)   # [K, 1, N, N]
        self.mul_L_imag = L_norm_imag.unsqueeze(1)   # [K, 1, N, N]

    def forward(self, data):
        """
        :param inputs: the input data, real [B, N, C], img [B, N, C]
        :param L_norm_real, L_norm_imag: the laplace, [N, N], [N,N]
        """
        X_real, X_imag = data[0], data[1]
        real = torch.matmul(self.mul_L_real, X_real)  # [K, B, N, C]
        real = torch.matmul(real, self.weight)  # [K, B, N, D]
        real = torch.sum(real, dim=0)  # [B, N, D]
        
        real_ = -1.0*torch.matmul(self.mul_L_imag, X_imag)  # [K, B, N, C]
        real_ = torch.matmul(real_, self.weight)  # [K, B, N, D]
        real_ = torch.sum(real_, dim=0)  # [B, N, D]

        imag = torch.matmul(self.mul_L_imag, X_real)  # [K, B, N, C]
        imag = torch.matmul(imag, self.weight)  # [K, B, N, D]
        imag = torch.sum(imag, dim=0)   # [B, N, D]

        imag_ = torch.matmul(self.mul_L_real, X_imag)  # [K, B, N, C]
        imag_ = torch.matmul(imag_, self.weight)  # [K, B, N, D]
        imag_ = torch.sum(imag_, dim=0)   # [B, N, D]

        return real+real_+self.bias, imag+imag_+ self.bias
    
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
    def __init__(self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim = 2, activation = False, layer = 2, to_radians = False, dropout = False):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super(ChebNet, self).__init__()

        chebs = [ChebConv(in_c=in_c, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag)]
        if activation and (layer != 1):
            chebs.append(complex_relu_layer())

        for i in range(1, layer):
            chebs.append(ChebConv(in_c=num_filter, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag))
            if activation:
                if (i == layer - 1) and (to_radians):
                    continue
                else:
                    chebs.append(complex_relu_layer())

        self.Chebs = torch.nn.Sequential(*chebs)
        self.to_radians = to_radians

        last_dim = 2
        if self.to_radians == 'combine':
            last_dim = 3
        self.Conv = nn.Conv1d(num_filter*last_dim, label_dim, kernel_size=1)        
        self.dropout = dropout

    def forward(self, real, imag):
        real, imag = self.Chebs((real, imag))
        if self.to_radians == 'radians':
            real[real==0.0] = 1e-8
            radians = torch.atan2(imag, real)
            modulus = torch.sqrt(torch.square(imag) + torch.square(real) + 1e-8)
            x = torch.cat((modulus, radians), dim = -1)
        elif self.to_radians == 'combine':
            real[real==0.0] = 1e-8
            radians = torch.atan2(imag, real)
            x = torch.cat((real, imag, radians), dim = -1)
        else:
            x = torch.cat((real, imag), dim = -1)
        
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = F.log_softmax(x, dim=1)
        return x

class ChebNet_Edge(nn.Module):
    def __init__(self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim = 2, activation = False, layer = 2, to_radians = False, dropout = False):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super(ChebNet_Edge, self).__init__()
        
        self.to_radians = to_radians
        chebs = [ChebConv(in_c=in_c, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag)]
        if activation and (layer != 1):
            chebs.append(complex_relu_layer())

        for i in range(1, layer):
            chebs.append(ChebConv(in_c=num_filter, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag))
            if activation:
                if (i == layer - 1) and (to_radians):
                    continue
                else:
                    chebs.append(complex_relu_layer())
        self.Chebs = torch.nn.Sequential(*chebs)
        
        last_dim = 2
        if self.to_radians == 'combine':
            last_dim = 3
        self.Conv = nn.Conv1d(num_filter*last_dim, label_dim, kernel_size=1)     
        self.dropout = dropout

    def forward(self, real, imag, positive, negative):
        real, imag = self.Chebs((real, imag))

        pos_real = real[:,positive[:,0]] - real[:,positive[:,1]]
        pos_imag = imag[:,positive[:,0]] - imag[:,positive[:,1]]

        neg_real = real[:,negative[:,0]] - real[:,negative[:,1]]
        neg_imag = imag[:,negative[:,0]] - imag[:,negative[:,1]]
        
        real = torch.cat((pos_real, neg_real), axis = 1)
        imag = torch.cat((pos_imag, neg_imag), axis = 1)

        if self.to_radians == 'radians':
            real[real==0.0] = 1e-8
            radians = torch.atan2(imag, real)
            modulus = torch.sqrt(torch.square(imag) + torch.square(real) + 1e-8)
            x = torch.cat((modulus, radians), dim = -1)
        elif self.to_radians == 'combine':
            real[real==0.0] = 1e-8
            radians = torch.atan2(imag, real)
            x = torch.cat((real, imag, radians), dim = -1)
        else:
            x = torch.cat((real, imag), dim = -1)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = x.permute((0,2,1))        
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()
        
        x = F.log_softmax(x, dim=1)
        return x