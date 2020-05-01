import os
import argparse
import time
import numpy as np

import dgl
from dgl.contrib import KVClient
from dgl import backend as F

num_entries = 1000*1000

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--ip_config', type=str, default='ip_config.txt',
                          help='IP configuration file of kvstore.')
        self.add_argument('--num_servers', type=int, default=2,
                          help='The number of servers')
        self.add_argument('--num_worker', type=int, default=1,
                          help='Number of worker (client nodes) on single-machine.')
        self.add_argument('--batch_size', type=int, default=1000,
                          help='The batch size')

def start_client(args):
    """Start client
    """
    server_namebook = dgl.contrib.read_ip_config(filename=args.ip_config)

    my_client = KVClient(server_namebook=server_namebook)

    my_client.connect()

    my_client.print()

    my_client.barrier()

    local_start = num_entries * my_client.get_machine_id()
    local_end = num_entries * (my_client.get_machine_id() + 1)
    local_range = np.arange(local_start, local_end)
    id_list = []
    for i in range(10000):
        ids = np.random.choice(local_range, args.batch_size)
        id_list.append(F.tensor(ids))

    print("Pull from local...")
    num_bytes = 0
    start = time.time()
    for ids in id_list:
        tmp = my_client.pull(name='entity_embed', id_tensor=ids)
        ndim = tmp.shape[1]
        num_bytes += np.prod(tmp.shape) * 4
    print("Total time: %.3f, #bytes: %.3f GB" % (time.time() - start, num_bytes / 1000 / 1000 / 1000))

    my_client.barrier()

    arr = F.zeros((num_entries, ndim), F.float32, F.cpu())
    print('Slice from a tensor...')
    num_bytes = 0
    start = time.time()
    for ids in id_list:
        tmp = arr[ids]
        num_bytes += np.prod(tmp.shape) * 4
    print("Total time: %.3f, #bytes: %.3f GB" % (time.time() - start, num_bytes / 1000 / 1000 / 1000))

    print("Pull from remote...")
    if local_start == 0:
        remote_range = np.arange(local_end, num_entries * args.num_servers)
    elif local_end == num_entries * args.num_servers:
        remote_range = np.arange(0, local_start)
    else:
        range1 = np.arange(0, local_start)
        range2 = np.arange(local_end, num_entries * args.num_servers)
        remote_range = np.concatenate((range1, range2))
    id_list = []
    for i in range(1000):
        ids = np.random.choice(remote_range, args.batch_size)
        id_list.append(F.tensor(ids))

    num_bytes = 0
    start = time.time()
    for ids in id_list:
        tmp = my_client.pull(name='entity_embed', id_tensor=ids)
        num_bytes += np.prod(tmp.shape) * 4
    print("Total time: %.3f, #bytes: %.3f GB" % (time.time() - start, num_bytes / 1000 / 1000 / 1000))

    my_client.barrier()

    if my_client.get_id() == 0:
        my_client.shut_down()


if __name__ == '__main__':
    args = ArgParser().parse_args()
    start_client(args)
