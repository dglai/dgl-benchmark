import dgl
import dgl.function as fn
import numpy as np
import scipy as sp
import torch as th

import time
import argparse

def torch_mm(adj, feat, use_gpu, warmup_steps, total_steps):
    if use_gpu:
        th.cuda.synchronize()
    accum_time = 0
    for cnt in range(total_steps):
        tic = time.time()
        y = adj @ feat
        if use_gpu:
            th.cuda.synchronize()    
        toc = time.time()
        if cnt >= warmup_steps:
            accum_time += toc - tic
    
    print('torch mm average speed {} ms'.format(accum_time / (total_steps - warmup_steps)))

def torch_spmm(adj, feat, use_gpu, warmup_steps, total_steps):
    if use_gpu:
        th.cuda.synchronize()
    accum_time = 0
    for cnt in range(total_steps):
        tic = time.time()
        y = adj @ feat
        if use_gpu:
            th.cuda.synchronize()    
        toc = time.time()
        if cnt >= warmup_steps:
            accum_time += toc - tic
    
    print('torch spmm average speed {} ms'.format(accum_time / (total_steps - warmup_steps)))

def dgl_kernel_fusion(g, use_gpu, warmup_steps, total_steps):
    if use_gpu:
        th.cuda.synchronize()
    accum_time = 0
    for cnt in range(total_steps):
        tic = time.time()
        g.update_all(fn.copy_u('x', 'm'), fn.sum('m', 'y'))
        y = g.ndata['y']
        if use_gpu:
            th.cuda.synchronize()    
        toc = time.time()
        if cnt >= warmup_steps:
            accum_time += toc - tic
    
    print('dgl kernel fusion average speed {} ms'.format(accum_time / (total_steps - warmup_steps)))

def prof_spmm(args):
    total_steps = 100
    warmup_steps = 10
    adj = sp.sparse.random(args.n, args.n, args.p)
    g = dgl.DGLGraph(adj)
    feat = th.rand(args.n, args.feat_size)
    device = th.device(0) if args.gpu else th.device('cpu')
    g.ndata['x'] = feat.to(device)
    adj = g.adjacency_matrix().to(device)
    dense = adj.to_dense()
    with th.no_grad():
        torch_mm(dense, feat, args.gpu, warmup_steps, total_steps)
        torch_spmm(adj, feat, args.gpu, warmup_steps, total_steps)
        dgl_kernel_fusion(g, args.gpu, warmup_steps, total_steps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SPMM benchmark')
    parser.add_argument('--feat-size', '-f', type=int, default=64)
    parser.add_argument('--n', '-n', type=int, default=1000)
    parser.add_argument('--p', '-p', type=float, default=0.01)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    prof_spmm(args)
    
