import os
import time
import numpy as np

import dgl
from dgl import backend as F

import argparse
import torch as th

def udf_pull(target, name, id_tensor):
    return target[name][id_tensor]

def create_partition_policy(args):
    """Create GraphPartitionBook and PartitionPolicy
    """
    g = dgl.DGLGraph()
    g.add_nodes(args.graph_size)
    g.add_edge(0, 1) # we don't use edge data in our benchmark

    global_nid = F.tensor(np.arange(args.graph_size) + args.machine_id * args.graph_size)
    global_eid = F.tensor([args.machine_id])

    node_map = np.zeros((args.graph_size*2), np.int64)
    node_map[args.graph_size:] = 1
    node_map = F.tensor(node_map)
    edge_map = F.tensor([0,1])

    g.ndata[dgl.NID] = global_nid
    g.edata[dgl.EID] = global_eid

    gpb = dgl.distributed.GraphPartitionBook(part_id=args.machine_id,
                                             num_parts=args.num_machine,
                                             node_map=node_map,
                                             edge_map=edge_map,
                                             part_graph=g)

    policy = dgl.distributed.PartitionPolicy(policy_str='node',
                                             part_id=args.machine_id,
                                             partition_book=gpb)
    return policy, gpb

def create_range_partition_policy(args):
    """Create RangePartitionBook and PartitionPolicy
    """
    node_map = F.tensor(np.array([args.graph_size, 2*args.graph_size], np.int64))
    edge_map = F.tensor([1,2])

    gpb = dgl.distributed.RangePartitionBook(part_id=args.machine_id,
                                             num_parts=args.num_machine,
                                             node_map=node_map,
                                             edge_map=edge_map)

    policy = dgl.distributed.PartitionPolicy(policy_str='node',
                                             part_id=args.machine_id,
                                             partition_book=gpb)
    return policy, gpb 

def create_data(args):
    """Create data hold by server nodes
    """
    data = F.zeros((args.graph_size, args.dim), F.float32, F.cpu())
    return data

def start_server(args):
    kvserver = dgl.distributed.KVServer(server_id=args.server_id,
                                        ip_config=args.ip_config,
                                        num_clients=args.num_client)

    if args.range == -1:
        policy, gpb = create_partition_policy(args)
    else:
        policy, gpb = create_range_partition_policy(args)

    data = create_data(args)
    kvserver.add_part_policy(policy)

    if kvserver.is_backup_server():
        kvserver.init_data(name='data', policy_str='node')
    else:
        kvserver.init_data(name='data', policy_str='node', data_tensor=data)

    server_state = dgl.distributed.ServerState(kvserver, None, None)

    dgl.distributed.start_server(server_id=args.server_id,
                                 ip_config=args.ip_config,
                                 num_clients=args.num_client,
                                 server_state=server_state)

def start_client(args):
    if args.range == -1:
        policy, gpb = create_partition_policy(args)
    else:
        policy, gpb = create_range_partition_policy(args)
    data = create_data(args)
    dgl.distributed.connect_to_server(ip_config=args.ip_config)
    kvclient = dgl.distributed.KVClient(ip_config=args.ip_config)
    kvclient.barrier()
    kvclient.map_shared_data(partition_book=gpb)

    #################################### local fast-pull ####################################

    
    if args.machine_id == 1:
        id_tensor = np.random.randint(args.graph_size, size=args.data_size)
        id_tensor = id_tensor + args.graph_size
    else:
        id_tensor = np.random.randint(args.graph_size, size=args.data_size)
    id_tensor = F.tensor(id_tensor)

    start = time.time()
    res = kvclient.pull(name='data', id_tensor=id_tensor)
    end = time.time()
    print("Total time of pull: %f" % (end-start))

    start = time.time()
    for _ in range(100):
        res = kvclient.pull(name='data', id_tensor=id_tensor)
    end = time.time()
    total_bytes = (args.data_size*(args.dim+2)*4)*100*args.num_client/2
    print("Local fast-pull Throughput (MB): %f" % (total_bytes / (end-start) / 1024.0 / 1024.0))
    
    

    #################################### remote fast-pull ####################################

    
    if args.machine_id == 0:
        id_tensor = np.random.randint(args.graph_size, size=args.data_size)
        id_tensor = id_tensor + args.graph_size
    else:
        id_tensor = np.random.randint(args.graph_size, size=args.data_size)
    id_tensor = F.tensor(id_tensor)

    start = time.time()
    for _ in range(100):
        res = kvclient.pull(name='data', id_tensor=id_tensor)
    end = time.time()
    total_bytes = (args.data_size*(args.dim+2)*4)*100*args.num_client/2
    print("Remote fast-pull Throughput (MB): %f" % (total_bytes / (end-start) / 1024.0 / 1024.0))
    
    

    #################################### local pull ##################################

    
    kvclient.register_pull_handler('data', udf_pull)
    kvclient.barrier()

    if args.machine_id == 1:
        id_tensor = np.random.randint(args.graph_size, size=args.data_size)
        id_tensor = id_tensor + args.graph_size
    else:
        id_tensor = np.random.randint(args.graph_size, size=args.data_size)
    id_tensor = F.tensor(id_tensor)

    start = time.time()
    for _ in range(100):
        res = kvclient.pull(name='data', id_tensor=id_tensor)
    end = time.time()
    total_bytes = (args.data_size*(args.dim+2)*4)*100*args.num_client/2
    print("Local pull Throughput (MB): %f" % (total_bytes / (end-start) / 1024.0 / 1024.0))
    

    #################################### remote pull ##################################

    
    if args.machine_id == 0:
        id_tensor = np.random.randint(args.graph_size, size=args.data_size)
        id_tensor = id_tensor + args.graph_size
    else:
        id_tensor = np.random.randint(args.graph_size, size=args.data_size)
    id_tensor = F.tensor(id_tensor)

    start = time.time()
    for _ in range(100):
        res = kvclient.pull(name='data', id_tensor=id_tensor)
    end = time.time()
    total_bytes = (args.data_size*(args.dim+2)*4)*100*args.num_client/2
    print("Remote pull Throughput (MB): %f" % (total_bytes / (end-start) / 1024.0 / 1024.0))
    
    
    ################################# local push ######################################

    
    if args.machine_id == 1:
        id_tensor = np.random.randint(args.graph_size, size=args.data_size)
        id_tensor = id_tensor + args.graph_size
    else:
        id_tensor = np.random.randint(args.graph_size, size=args.data_size)
    id_tensor = F.tensor(id_tensor)
    data_tensor = F.zeros((args.data_size, args.dim), F.float32, F.cpu())

    kvclient.barrier()
    start = time.time()
    for _ in range(100):
        res = kvclient.push(name='data', id_tensor=id_tensor, data_tensor=data_tensor)
    kvclient.barrier()
    end = time.time()
    total_bytes = (args.data_size*(args.dim+2)*4)*100*args.num_client/2
    print("Local push Throughput (MB): %f" % (total_bytes / (end-start) / 1024.0 / 1024.0))
    

    ################################# remote push ######################################

    
    if args.machine_id == 0:
        id_tensor = np.random.randint(args.graph_size, size=args.data_size)
        id_tensor = id_tensor + args.graph_size
    else:
        id_tensor = np.random.randint(args.graph_size, size=args.data_size)
    id_tensor = F.tensor(id_tensor)

    kvclient.barrier()
    start = time.time()
    for _ in range(100):
        res = kvclient.push(name='data', id_tensor=id_tensor, data_tensor=data_tensor)
    kvclient.barrier()
    end = time.time()
    total_bytes = (args.data_size*(args.dim+2)*4)*100*args.num_client/2
    print("Remote push Throughput (MB): %f" % (total_bytes / (end-start) / 1024.0 / 1024.0))
    

    dgl.distributed.shutdown_servers()
    dgl.distributed.finalize_client()

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--range', type=int, default=1,
                          help='If use range partition.')
        self.add_argument('--num_client', type=int, default=2,
                          help='Total number of clients.')
        self.add_argument('--num_machine', type=int, default=2,
                          help="number of machine.")
        self.add_argument('--machine_id', type=int, default=0,
                          help="machine ID.")
        self.add_argument('--server_id', type=int, default=-1,
                          help='server_id')
        self.add_argument('--ip_config', type=str, default='ip_config.txt',
                          help='IP configuration file of kvstore.')
        self.add_argument('--data_size', type=int, default=100000,
                          help='data_size of each machine.')
        self.add_argument('--dim', type=int, default=10,
                          help='dim of each data.')
        self.add_argument('--graph_size', type=int, default=1000000,
                          help='total size of the graph.')
        self.add_argument('--threads', type=int, default=-1,
                          help='number of pytorch threads.')

if __name__ == '__main__':
    args = ArgParser().parse_args()

    if args.threads != -1:
        th.set_num_threads(args.threads)

    if args.server_id == -1:
        time.sleep(2)
        start_client(args)
    else:
        start_server(args)