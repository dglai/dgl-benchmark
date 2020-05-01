import os
import argparse
import time
import numpy as np

import dgl
from dgl.contrib import KVServer
from dgl import backend as F

num_entries = 1000*1000

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--server_id', type=int, default=0,
                          help='Unique ID of each server.')
        self.add_argument('--num_servers', type=int, default=2,
                          help='The number of servers.')
        self.add_argument('--ip_config', type=str, default='ip_config.txt',
                          help='IP configuration file of kvstore.')
        self.add_argument('--num_client', type=int, default=1,
                          help='Total number of client nodes.')
        self.add_argument('--dim_size', type=int, default=40,
                          help='The dimension size of the data.')


def start_server(args):
    """Start kvstore service
    """
    server_namebook = dgl.contrib.read_ip_config(filename=args.ip_config)

    my_server = KVServer(server_id=args.server_id, server_namebook=server_namebook, num_client=args.num_client)

    data = F.zeros((num_entries, args.dim_size), F.float32, F.cpu())
    g2l = F.zeros(num_entries * args.num_servers, F.int64, F.cpu())
    start = num_entries * my_server.get_machine_id()
    end = num_entries * (my_server.get_machine_id() + 1)
    g2l[start:end] = F.arange(0, num_entries)

    partition = np.arange(args.num_servers)
    partition = F.tensor(np.repeat(partition, num_entries))
    if my_server.get_id() % my_server.get_group_count() == 0: # master server
        my_server.set_global2local(name='entity_embed', global2local=g2l)
        my_server.init_data(name='entity_embed', data_tensor=data)
        my_server.set_partition_book(name='entity_embed', partition_book=partition)
    else:
        my_server.set_global2local(name='entity_embed')
        my_server.init_data(name='entity_embed')
        my_server.set_partition_book(name='entity_embed')


    my_server.print()

    my_server.start()
    

if __name__ == '__main__':
    args = ArgParser().parse_args()
    start_server(args)
