'''
Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

This file is part of the implementation as described in the NIPS 2018 paper:
Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
Please see the file LICENSE.txt for the license governing this code.
'''

import math

import torch.nn as nn

# from . import non_local
import math
from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import ops_integrated as ops

def convnxn(in_planes, out_planes, kernelsize, stride=1, bias=False):
    padding = kernelsize//2
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, padding=padding, bias=bias)

def dncnn_batchnorm_init(m, kernelsize=3, b_min=0.025):
    r"""
    Reproduces batchnorm initialization from DnCNN
    https://github.com/cszn/DnCNN/blob/master/TrainingCodes/DnCNN_TrainingCodes_v1.1/DnCNN_init_model_64_25_Res_Bnorm_Adam.m
    """
    n = kernelsize**2 * m.num_features
    m.weight.data.normal_(0, math.sqrt(2. / (n)))
    m.weight.data[(m.weight.data > 0) & (m.weight.data <= b_min)] = b_min
    m.weight.data[(m.weight.data < 0) & (m.weight.data >= -b_min)] = -b_min
    m.weight.data = m.weight.data.abs()
    m.bias.data.zero_()
    m.momentum = 0.001


class DnCNN(nn.Module):
    r"""
    Implements a DnCNN network
    """
    def __init__(self,  nplanes_in, nplanes_out, features, kernel, depth, residual, bn):
        r"""
        :param nplanes_in: number of of input feature channels
        :param nplanes_out: number of of output feature channels
        :param features: number of of hidden layer feature channels
        :param kernel: kernel size of convolution layers
        :param depth: number of convolution layers (minimum 2)
        :param residual: whether to add a residual connection from input to output
        :param bn:  whether to add batchnorm layers
        """
        super(DnCNN, self).__init__()

        self.residual = residual
        self.nplanes_out = nplanes_out
        self.nplanes_in = nplanes_in
        self.kernelsize = kernel
        self.nplanes_residual = None

        self.conv1 = convnxn(nplanes_in, features, kernelsize=kernel, bias=True)
        self.bn1 = nn.BatchNorm2d(features) if bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        layers = []
        for i in range(depth-2):
            layers += [convnxn(features, features, kernel),
                       nn.BatchNorm2d(features)  if bn else nn.Sequential(),
                       self.relu]
        self.layer1 = nn.Sequential(*layers)
        self.conv2 = convnxn(features , nplanes_out, kernelsize=kernel, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / (n)))
            elif isinstance(m, nn.BatchNorm2d):
                dncnn_batchnorm_init(m, kernelsize=self.kernelsize, b_min=0.025)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.conv2(x)

        nplanes_residual = self.nplanes_residual or self.nplanes_in
        if self.residual:
            nshortcut = min(self.nplanes_in, self.nplanes_out, nplanes_residual)
            x[:,:nshortcut,:,:] = x[:,:nshortcut,:,:] + shortcut[:,:nshortcut,:,:]

        return x


def cnn_from_def(cnn_opt):
    kernel = cnn_opt.get("kernel",3)
    padding = (kernel-1)//2
    cnn_bn = cnn_opt.get("bn",True)
    cnn_depth = cnn_opt.get("depth",0)
    cnn_channels = cnn_opt.get("features")
    cnn_outchannels = cnn_opt.get("nplanes_out",)
    chan_in = cnn_opt.get("nplanes_in")

    if cnn_depth == 0:
        cnn_outchannels=chan_in

    cnn_layers = []
    relu = nn.ReLU(inplace=True)

    for i in range(cnn_depth-1):
        cnn_layers.extend([
            nn.Conv2d(chan_in,cnn_channels,kernel, 1, padding, bias=not cnn_bn),
            nn.BatchNorm2d(cnn_channels) if cnn_bn else nn.Sequential(),
            relu
        ])
        chan_in = cnn_channels

    if cnn_depth > 0:
        cnn_layers.append(
            nn.Conv2d(chan_in,cnn_outchannels,kernel, 1, padding, bias=True)
        )

    net = nn.Sequential(*cnn_layers)
    net.nplanes_out = cnn_outchannels
    net.nplanes_in = cnn_opt.get("nplanes_in")
    return net


class N3Block(nn.Module):
    r"""
    N3Block operating on a 2D images
    """
    def __init__(self, nplanes_in, k, patchsize=10, stride=5,
                 nl_match_window=3,
                 temp_opt={}, embedcnn_opt={}):
        r"""
        :param nplanes_in: number of input features
        :param k: number of neighbors to sample
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        :param nl_match_window: size of matching window around each patch,
            i.e. the nl_match_window x nl_match_window patches around a query patch
            are used for matching
        :param temp_opt: options for handling the the temperature parameter
        :param embedcnn_opt: options for the embedding cnn, also shared by temperature cnn
        """
        super(N3Block, self).__init__()
        self.patchsize = patchsize
        self.stride = stride

        # patch embedding
        embedcnn_opt["nplanes_in"] = nplanes_in
        self.embedcnn = cnn_from_def(embedcnn_opt)

        # temperature cnn
        with_temp = temp_opt.get("external_temp")
        if with_temp:
            tempcnn_opt = dict(**embedcnn_opt)
            tempcnn_opt["nplanes_out"] = 1
            self.tempcnn = cnn_from_def(tempcnn_opt)
        else:
            self.tempcnn = None

        self.nplanes_in = nplanes_in
        self.nplanes_out = (k+1) * nplanes_in

        indexer = lambda xe_patch,ye_patch: index_neighbours(xe_patch, ye_patch, nl_match_window, exclude_self=True)
        self.n3aggregation = N3Aggregation2D(indexing=indexer, k=k,
                patchsize=patchsize, stride=stride, temp_opt=temp_opt)
        self.k = k

        self.reset_parameters()

    def forward(self, x):
        if self.k <= 0:
            return x

        xe = self.embedcnn(x)
        ye = xe

        xg = x
        if self.tempcnn is not None:
            log_temp = self.tempcnn(x)
        else:
            log_temp = None

        x = self.n3aggregation(xg,xe,ye,log_temp=log_temp)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d)):
                dncnn_batchnorm_init(m, kernelsize=3, b_min=0.025)

class N3Net(nn.Module):
    r"""
    A N3Net interleaves DnCNNS for local processing with N3Blocks for non-local processing
    """
    def __init__(self, nplanes_in, nplanes_out, nplanes_interm, nblocks, block_opt, nl_opt, residual=False):
        r"""
        :param nplanes_in: number of input features
        :param nplanes_out: number of output features
        :param nplanes_interm: number of intermediate features, i.e. number of output features for the DnCNN sub-networks
        :param nblocks: number of DnCNN sub-networks
        :param block_opt: options passed to DnCNNs
        :param nl_opt: options passed to N3Blocks
        :param residual: whether to have a global skip connection
        """
        super(N3Net, self).__init__()
        self.nplanes_in = nplanes_in
        self.nplanes_out = nplanes_out
        self.nblocks = nblocks
        self.residual = residual

        nin = nplanes_in
        cnns = []
        nls = []
        for i in range(nblocks-1):
            cnns.append(DnCNN(nin, nplanes_interm, **block_opt))
            nl = N3Block(nplanes_interm, **nl_opt)
            nin = nl.nplanes_out
            nls.append(nl)

        nout = nplanes_out
        cnns.append(DnCNN(nin, nout, **block_opt))

        self.nls = nn.Sequential(*nls)
        self.blocks = nn.Sequential(*cnns)

    def forward(self, x):
        shortcut = x
        for i in range(self.nblocks-1):
            x = self.blocks[i](x)
            x = self.nls[i](x)

        x = self.blocks[-1](x)

        if self.residual:
            nshortcut = min(self.nplanes_in, self.nplanes_out)
            x[:,:nshortcut,:,:] = x[:,:nshortcut,:,:] + shortcut[:,:nshortcut,:,:]

        return x

r"""
Shape parameters: B -> batch size, N -> number of database items
    M -> number of query items, F -> feature channels of database/query items
    E -> feature channels of embedding
    O -> number of potential neighbors per query item
"""

def compute_distances(xe, ye, I, train=True):
    r"""
    Computes pairwise distances for all pairs of query items and
    potential neighbors.

    :param xe: BxNxE tensor of database item embeddings
    :param ye: BxMxE tensor of query item embeddings
    :param I: BxMxO index tensor that selects O potential neighbors for each item in ye
    :param train: whether to use tensor comprehensions for inference (forward only)

    :return: a BxMxO tensor of distances
    """
    device = 'cuda:0'
    # xe -> b n e
    # ye -> b m e
    # I  -> b m o
    b,n, e = xe.shape
    m = ye.shape[1]
    o = I.shape[2]

    if not train:
        # xe_ind -> b m o e
        If = I.view(b, m*o,1).expand(b,m*o,e)

        # D -> b m o
        ye = ye.unsqueeze(3)

        D = -2*ops.indexed_matmul_1_efficient(xe, ye.squeeze(3), I).unsqueeze(3)

        xe_sqs = (xe**2).sum(dim=-1, keepdim=True)
        xe_sqs_ind = xe_sqs.gather(dim=1, index=If[:,:,0:1]).view(b,m,o,1)
        D += xe_sqs_ind
        D += (ye**2).sum(dim=-2, keepdim=True)

        D = D.squeeze(3)
    else:
        # D_full -> b m n
        D_full = ops.euclidean_distance(ye, xe.permute(0,2,1)).to(device)

        # D -> b m o
        D = torch.gather(D_full,dim=2, index=I)

    return -D

def aggregate_output(W,x,I, train=True):
    r"""
    Calculates weighted averages for k nearest neighbor volumes.

    :param W: BxMxOxK matrix of weights
    :param x: BxNxF tensor of database items
    :param I: BxMxO index tensor that selects O potential neighbors for each item in ye
    :param train: whether to use tensor comprehensions for inference (forward only)

    :return: a BxMxFxK tensor of the k nearest neighbor volumes for each query item
    """

    # W -> b m o k
    # x -> b n f
    # I -> b m o
    b,n,f = x.shape
    m,o = I.shape[1:3]
    k = W.shape[3]
    # print(b,m,o,k,f,n)

    z = ops.indexed_matmul_2_efficient(x, W,I)

    return z


def log1mexp(x, expm1_guard = 1e-7):
    # See https://cran.r-project.org/package=Rmpfr/.../log1mexp-note.pdf
    t = x < math.log(0.5)
    y = torch.zeros_like(x)
    y[t] = torch.log1p(-x[t].exp())

    # for x close to 0 we need expm1 for numerically stable computation
    # we furtmermore modify the backward pass to avoid instable gradients,
    # ie situations where the incoming output gradient is close to 0 and the gradient of expm1 is very large
    expxm1 = torch.expm1(x[~t])
    log1mexp_fw = (-expxm1).log()
    log1mexp_bw = (-expxm1+expm1_guard).log() # limits magnitude of gradient

    y[~t] = log1mexp_fw.detach() + (log1mexp_bw - log1mexp_bw.detach())
    return y



class NeuralNearestNeighbors(nn.Module):
    r"""
    Computes neural nearest neighbor volumes based on pairwise distances
    """
    def __init__(self, k, temp_opt={}):
        r"""
        :param k: Number of neighbor volumes to compute
        :param temp_opt: temperature options:
            external_temp: Whether temperature is given as external input
                rather than fixed parameter
            temp_bias: A fixed bias to add to the log temperature
            distance_bn: Whether to put distances through a batchnorm layer
        """
        super(NeuralNearestNeighbors, self).__init__()
        self.external_temp = temp_opt.get("external_temp")
        self.log_temp_bias = log(temp_opt.get("temp_bias", 1))
        distance_bn = temp_opt.get("distance_bn")

        if not self.external_temp:
            self.log_temp = nn.Parameter(torch.FloatTensor(1).fill_(0.0))
        if distance_bn:
            self.bn = nn.BatchNorm1d(1)
        else:
            self.bn = None

        self.k = k

    def forward(self, D, log_temp=None):
        b,m,o = D.shape
        if self.bn is not None:
            D = self.bn(D.view(b,1,m*o)).view(D.shape)

        if self.external_temp:
            log_temp = log_temp.view(D.shape[0], D.shape[1], -1)
        else:
            log_temp = self.log_temp.view(1,1,1)

        log_temp = log_temp + self.log_temp_bias

        temperature = log_temp.exp()
        if self.training:
            M = D.data > -float("Inf")
            if len(temperature) > 1:
                D[M] /= temperature.expand_as(D)[M]
            else:
                D[M] = D[M] / temperature[0,0,0]
        else:
            D /= temperature

        logits = D.view(D.shape[0]*D.shape[1],-1)


        samples_arr = []

        for r in range(self.k):
            # Eqs. 8 and 10
            weights = F.log_softmax(logits, dim=1)
            # weights_exp = ops.clamp_probs(weights.exp())
            weights_exp = weights.exp()

            samples_arr.append(weights_exp.view(b,m,o))

            # Eq. 9
            logits = logits + log1mexp(weights.view(*logits.shape))
            # logits = logits + (1-weights_exp.view(*logits.shape)).log()

        W = torch.stack(samples_arr,dim=3)

        return W

class N3AggregationBase(nn.Module):
    r"""
    Domain agnostic base class for computing neural nearest neighbors
    """
    def __init__(self, k, temp_opt={}):
        r"""
        :param k: Number of neighbor volumes to compute
        :param temp_opt: options for handling temperatures, see `NeuralNearestNeighbors`
        """
        super(N3AggregationBase, self).__init__()
        self.k = k
        self.nnn = NeuralNearestNeighbors(k, temp_opt=temp_opt)

    def forward(self, x, xe, ye, I, log_temp=None):
        r"""
        :param x: database items, shape BxNxF
        :param xe: embedding of database items, shape BxNxE
        :param ye: embedding of query items, shape BxMxE
        :param y: query items, if None then y=x is assumed, shape BxMxF
        :param I: Indexing tensor defining O potential neighbors for each query item
            shape BxMxO
        :param log_temp: optional log temperature
        :return:
        """

        # x  -> b n f
        # xe -> b n e
        # ye -> b m e
        # I  -> b m o
        b, n, f = x.shape
        m, e = ye.shape[1:]
        o = I.shape[2]
        k = self.k

        assert((b,n,e) == xe.shape)
        assert((b,m,e) == ye.shape)
        assert((b,m,o) == I.shape)

        # compute distance
        D = compute_distances(xe, ye, I, train=self.training)
        assert((b,m,o) == D.shape)

        # compute aggregation weights
        W = self.nnn(D, log_temp=log_temp)

        assert((b,m,o,k) == W.shape)
        # aggregate output
        z = aggregate_output(W, x, I, train=self.training)
        assert((b,m,f,k) == z.shape)

        return z

class N3Aggregation2D(nn.Module):
    r"""
    Computes neural nearest neighbors for image data based on extracting patches
    in strides.
    """
    def __init__(self, indexing, k, patchsize, stride, temp_opt={}, padding=None):
        r"""
        :param indexing: function for creating index tensor
        :param k: number of neighbor volumes
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        :param temp_opt: options for handling temperatures, see `NeuralNearestNeighbors`
        """
        super(N3Aggregation2D, self).__init__()
        self.patchsize = patchsize
        self.stride = stride
        self.indexing = indexing
        self.k = k
        self.temp_opt = temp_opt
        self.padding = padding
        if k <= 0:
            self.aggregation = None
        else:
            self.aggregation = N3AggregationBase(k, temp_opt=temp_opt)

    def forward(self, x, xe, ye, y=None, log_temp=None):
        r"""
        :param x: database image
        :param xe: embedding of database image
        :param ye: embedding of query image
        :param y: query image, if None then y=x is assumed
        :param log_temp: optional log temperature image
        :return:
        """
        if self.aggregation is None:
            return y if y is not None else x

        # Convert everything to patches
        x_patch, padding = ops.im2patch(x, self.patchsize, self.stride, None, returnpadding=True)
        xe_patch = ops.im2patch(xe, self.patchsize, self.stride, self.padding)
        if y is None:
            y = x
            ye_patch = xe_patch
        else:
            ye_patch = ops.im2patch(ye, self.patchsize, self.stride, self.padding)

        I = self.indexing(xe_patch, ye_patch)
        if not self.training:
            index_neighbours_cache.clear()

        b,c,p1,p2,n1,n2 = x_patch.shape
        _,ce,e1,e2,m1,m2 = ye_patch.shape
        _,_,o = I.shape
        k = self.k
        _,_,H,W = y.shape
        n = n1*n2; m=m1*m2; f=c*p1*p2; e=ce*e1*e2

        x_patch = x_patch.permute(0,4,5,1,2,3).contiguous().view(b,n,f)
        xe_patch = xe_patch.permute(0,4,5,1,2,3).contiguous().view(b,n,e)
        ye_patch = ye_patch.permute(0,4,5,1,2,3).contiguous().view(b,m,e)

        if log_temp is not None:
            log_temp_patch = ops.im2patch(log_temp, self.patchsize, self.stride, self.padding)
            log_temp_patch = log_temp_patch.permute(0,4,5,2,3,1).contiguous().view(b,m,self.patchsize**2, log_temp.shape[1])
            if self.temp_opt["avgpool"]:
                log_temp_patch = log_temp_patch.mean(dim=2)
            else:
                log_temp_patch = log_temp_patch[:,:,log_temp_patch.shape[2]//2,:].contiguous()
        else:
            log_temp_patch = None

        # Get nearest neighbor volumes
        # z  -> b m1*m2 c*p1*p2 k
        z_patch = self.aggregation(x_patch, xe_patch, ye_patch, I, log_temp=log_temp_patch)
        z_patch = z_patch.permute(0,1,3,2).contiguous().view(b,m1,m2,k*c,p1,p2).permute(0,3,4,5,1,2).contiguous()

        # Convert patches back to whole images
        z = ops.patch2im(z_patch, self.patchsize, self.stride, padding)

        z = z.contiguous().view(b,k,c,H,W)
        z = z-y.view(b,1,c,H,W)
        z = z.view(b,k*c,H,W)

        # Concat with input
        z = torch.cat([y, z], dim=1)

        return z


index_neighbours_cache = {}
def index_neighbours(xe_patch, ye_patch, s, exclude_self=True):
    r"""
    This function generates the indexing tensors that define neighborhoods for each query patch
    It selects a neighborhood of s x s patches around each patch.
    Index tensors get cached in order to speed up execution time. This might lead to
    memory problems, though.
    """
    o = s**2
    if exclude_self:
        o-=1
    b,_,_,_,n1,n2 = xe_patch.shape
    n = n1*n2
    b,_,_,_,m1,m2 = ye_patch.shape
    m = m1*m2

    assert(m==n)

    dev = xe_patch.get_device()
    key = "{}_{}_{}_{}_{}_{}_{}".format(n1,n2,m1,m2,s,exclude_self, dev)
    if not key in index_neighbours_cache:
        I = torch.empty(1,m1*m2,o, device=dev, dtype=torch.int64)

        ih = torch.tensor(range(s), device=dev, dtype=torch.int64).view(1,1,s,1)
        iw = torch.tensor(range(s), device=dev, dtype=torch.int64).view(1,1,1,s)*n2

        i = torch.tensor(range(m1), device=dev, dtype=torch.int64).view(m1,1,1,1)
        j = torch.tensor(range(m2), device=dev, dtype=torch.int64).view(1,m2,1,1)

        ch = (i-s//2).clamp(0,n1-s)
        cw = (j-s//2).clamp(0,n2-s)

        cidx = ch*n2+cw
        midx = (i*m2+j).view(m1,m2,1)

        mI = cidx + ih + iw
        mI = mI.view(m1,m2,-1)
        
        mI = mI[mI!=midx][:(mI[mI!=midx].shape[0]//(m1*m2))*(m1*m2)].view(m1*m2,-1)
        
        I[0,:,:mI.shape[1]] = mI
        # I2 = I.clone()

        # ih = torch.LongTensor(range(s)).view(s,1)
        # iw = torch.LongTensor(range(s)).view(1,s)*n2
        # for i in range(m1):
        #     for j in range(m2):
        #         midx = i*m2+j
        #
        #         ch = min(n1-s, max(0,i-s//2))
        #         cw = min(n2-s, max(0,j-s//2))
        #         cidx = ch*n2+cw
        #         mI = cidx + ih + iw
        #         mI = mI.view(-1)
        #         if exclude_self:
        #             mI = mI[mI!=midx]
        #
        #         I[0,midx,:] = mI
        #
        # ih = ih.cuda(dev)
        # iw = iw.cuda(dev)
        index_neighbours_cache[key] = I

    I = index_neighbours_cache[key]
    I = I.repeat(b,1,1)
    return Variable(I, requires_grad=False)
