import networkx as nx
from dgl.data.citation_graph import GCNSyntheticDataset

def load_synthetic(args):
    n = args.n_nodes
    m = args.degree
    def gen(seed):
        return nx.barabasi_albert_graph(n, m, seed)
    return GCNSyntheticDataset(gen,
                               args.syn_nfeats,
                               args.syn_nclasses,
                               args.syn_train_ratio,
                               args.syn_val_ratio,
                               args.syn_test_ratio,
                               args.syn_seed)
