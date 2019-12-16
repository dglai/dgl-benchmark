import dgl
import dgl.function as fn
import numpy as np
import scipy as sp
import torch as th

import time
from itertools import product

D1 = 64
D2 = 64
D_EBED = 128

def add(a, b):
    return a + b

def sub(a, b):
    return a - b

def mul(a, b):
    return a * b

def div(a, b):
    return a / b

def sum(x, dim):
    return x.sum(dim)

def max(x, dim):
    return x.max(dim)[0]

def min(x, dim):
    return x.min(dim)[0]

def mean(x, dim):
    return x.min(dim)[0]

def prod(x, dim):
    return x.prod(dim)

def matmul(a, b):
    return a @ b

def dot(a, b):
    return sum(mul(a, b), dim=-1)

udf_ops = {
    'add': add,
    'sub': sub,
    'mul': mul,
    'div': div,
    'sum': sum,
    'max': max,
    'min': min,
    'mean': mean,
    'prod': prod,
    'matmul': matmul,
    'dot': dot,
}

def generate_feature(g, broadcast='none', binary_op='none', level=2):
    """Create graph with src, edge, dst feature. broadcast can be 'u',
    'e', 'v', 'none'
    """
    nv = g.number_of_nodes()
    ne = g.number_of_edges()
    if level == 1:
        if binary_op == 'dot':
            if broadcast == 'e':
                u = th.randn((nv, D1, D_EBED))
                e = th.randn((ne, 1, D_EBED))
                v = th.randn((nv, D1, D_EBED))
            elif broadcast == 'u':
                u = th.randn((nv, 1, D_EBED))
                e = th.randn((ne, D1, D_EBED))
                v = th.randn((nv, D1, D_EBED))
            elif broadcast == 'v':
                u = th.randn((nv, D1, D_EBED))
                e = th.randn((ne, D1, D_EBED))
                v = th.randn((nv, 1, D_EBED))
            else:
                u = th.randn((nv, D1, D_EBED))
                e = th.randn((ne, D1, D_EBED))
                v = th.randn((nv, D1, D_EBED))
        else:
            if broadcast == 'e':
                u = th.randn((nv, D1))
                e = th.randn((ne, 1))
                v = th.randn((nv, D1))
            elif broadcast == 'u':
                u = th.randn((nv, 1))
                e = th.randn((ne, D1))
                v = th.randn((nv, D1))
            elif broadcast == 'v':
                u = th.randn((nv, D1))
                e = th.randn((ne, D1))
                v = th.randn((nv, 1))
            else:
                u = th.randn((nv, D1))
                e = th.randn((ne, D1))
                v = th.randn((nv, D1))
    else:
        if binary_op == 'dot':
            if broadcast == 'e':
                u = th.randn((nv, D1, D2, D_EBED))
                e = th.randn((ne, D1, 1, D_EBED))
                v = th.randn((nv, D1, D2, D_EBED))
            elif broadcast == 'u':
                u = th.randn((nv, D1, 1, D_EBED))
                e = th.randn((ne, D1, D2, D_EBED))
                v = th.randn((nv, D1, D2, D_EBED))
            elif broadcast == 'v':
                u = th.randn((nv, D1, D2, D_EBED))
                e = th.randn((ne, D1, D2, D_EBED))
                v = th.randn((nv, D2, 1, D_EBED))
            else:
                u = th.randn((nv, D1, D2, D_EBED))
                e = th.randn((ne, D1, D2, D_EBED))
                v = th.randn((nv, D1, D2, D_EBED))
        else:
            if broadcast == 'e':
                u = th.randn((nv, D1, D2))
                e = th.randn((ne, 1, D2))
                v = th.randn((nv, D1, D2))
            elif broadcast == 'u':
                u = th.randn((nv, 1, D2))
                e = th.randn((ne, D1, D2))
                v = th.randn((nv, D1, D2))
            elif broadcast == 'v':
                u = th.randn((nv, D1, D2))
                e = th.randn((ne, D1, D2))
                v = th.randn((nv, 1, D2))
            else:
                u = th.randn((nv, D1, D2))
                e = th.randn((ne, D1, D2))
                v = th.randn((nv, D1, D2))
    return u, v, e


def prof_binary_builtins():
    def _prof(g, lhs, rhs, binary_op, reducer, broadcast='none', level=1, dev='cuda:0'):
        hu, hv, he = generate_feature(g, broadcast, binary_op, level)

        g.ndata['u'] = hu.requires_grad_()
        g.ndata['v'] = hv.requires_grad_()
        g.edata['e'] = he.requires_grad_()

        g.to(th.device(dev))

        builtin_msg_name = "{}_{}_{}".format(lhs, binary_op, rhs)
        builtin_msg = getattr(fn, builtin_msg_name)
        builtin_red = getattr(fn, reducer)

        def target_feature_switch(g, target):
            if target == "u":
                return g.ndata["u"]
            elif target == "v":
                return g.ndata["v"]
            else:
                return g.edata["e"]

        def time_forward_backward():
            dur_f = []
            dur_b = []
            for i in range(50):
                if i >= 5:
                    t0 = time.time()
                g.update_all(builtin_msg(lhs, rhs, 'm'), builtin_red('m', 'r1'))
                r1 = g.ndata.pop('r1')
                if i >= 5:
                    dur_f.append(time.time() - t0)

                grad = r1.sum()

                if i >= 5:
                    t0 = time.time()
                grad.backward()
                if i >= 5:
                    dur_b.append(time.time() - t0)
            return np.average(dur_f), np.average(dur_b)

        builtin_forward_time, builtin_backward_time = time_forward_backward()
        def target_switch(edges, target):
            if target == "u":
                return edges.src
            elif target == "v":
                return edges.dst
            elif target == "e":
                return edges.data
            else:
                assert(0), "Unknown target {}".format(target)

        def mfunc(edges):
            op = udf_ops[binary_op]
            lhs_data = target_switch(edges, lhs)[lhs]
            rhs_data = target_switch(edges, rhs)[rhs]

            while lhs_data.dim() < rhs_data.dim():
                lhs_data = F.unsqueeze(lhs_data, 1)
            while rhs_data.dim() < lhs_data.dim():
                rhs_data = F.unsqueeze(rhs_data, 1)
            return {"m": op(lhs_data, rhs_data)}

        def rfunc(nodes):
            op = udf_ops[reducer]
            return {"r2": op(nodes.mailbox['m'], 1)}

        def udf_time_forward_backward():
            dur_f = []
            dur_b = []
            for i in range(50):
                if i >= 5:
                    t0 = time.time()
                g.update_all(mfunc, rfunc)
                r2 = g.ndata.pop('r2')
                if i >= 5:
                    dur_f.append(time.time() - t0)

                grad = r2.sum()

                if i >= 5:
                    t0 = time.time()
                grad.backward()
                if i >= 5:
                    dur_b.append(time.time() - t0)
            return np.average(dur_f), np.average(dur_b)

        udf_forward_time, udf_backward_time = udf_time_forward_backward()
        print("--------------------------------------------------------------")
        print("{}_{} | {} | {}_{}_{} | {} | {} | {:06.5f} | {:06.5f} | {:06.5f} | {:06.5f} ".
            format(D1, D2, binary_op, lhs, binary_op, rhs, reducer, broadcast,
                udf_forward_time.astype(float), builtin_forward_time.astype(float),
                udf_backward_time.astype(float), builtin_backward_time.astype(float)))

    adj = sp.sparse.random(1000, 1000, 0.05)
    G = dgl.DGLGraph(adj)
    target = ["u", "v", "e"]
    for size in [64, 128, 256]:
        D1 = size
        for level in [1,2]:
            for lhs, rhs in product(target, target):
                if lhs == rhs:
                    continue
                for binary_op in ["add", "sub", "mul", "div", "dot"]:
                    for reducer in ["sum", "max", "min", "mean"]:
                        for broadcast in ["none", lhs, rhs]:
                            _prof(G, lhs, rhs, binary_op, reducer, broadcast=broadcast)

if __name__ == '__main__':
    prof_binary_builtins()
