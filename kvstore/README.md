# Distributed kvstore benchmark

Environment: two EC2 `r5dn.24xlarge` instances, `48` cores and `100` Gbit network

As a example, we run 2 server and 2 client on each machine.

1. Change `ip_config.txt` to set the ip address and server count to `2` like this:

    172.31.40.126 30050 2
    172.31.36.228 30050 2

2. Start server nodes on each machine:

machine-0: 

    python3 benchmark.py --server_id 0 --machine_id 0 &
    python3 benchmark.py --server_id 1 --machine_id 0 &

machine-1:

    python3 benchmark.py --server_id 2 --machine_id 1 &
    python3 benchmark.py --server_id 3 --machine_id 1 &
    
3. Start client nodes on each machine:

machine-0:

    python3 benchmark.py --machine_id 0 &
    python3 benchmark.py --machine_id 0 &

machine-1:

    python3 benchmark.py --machine_id 1 &
    python3 benchmark.py --machine_id 1 &
    
You can change the arguments like `data_size`, `dim`, as well as `threads` in the program.

Benchmark result:
==================

1 client & 1 server

100k * 10

Local fast-pull Throughput (MB): 1309.554908
Remote fast-pull Throughput (MB): 555.820761
Local pull Throughput (MB): 569.412073
Remote pull Throughput (MB): 334.965301
Local push Throughput (MB): 1102.261761
Remote push Throughput (MB): 804.207345

100k * 200

Local fast-pull Throughput (MB): 6817.679390
Remote fast-pull Throughput (MB): 589.524837
Local pull Throughput (MB): 2643.266993
Remote pull Throughput (MB): 698.920039
Local push Throughput (MB): 4831.602507
Remote push Throughput (MB): 877.408615

———————————

4 client & 4 server

100k * 10

Local fast-pull Throughput (MB): 5271.720105
Remote fast-pull Throughput (MB): 1664.377786
Local pull Throughput (MB): 1846.861070
Remote pull Throughput (MB): 1131.571235
Local push Throughput (MB): 2737.799059
Remote push Throughput (MB): 2962.085308

100k * 200

Local fast-pull Throughput (MB): 15671.273699
Remote fast-pull Throughput (MB): 1967.417778
Local pull Throughput (MB): 6180.696644
Remote pull Throughput (MB): 2234.634593
Local push Throughput (MB): 10667.588198
Remote push Throughput (MB): 3791.462856

—————————————

20 client & 20 server

100k * 10

Local fast-pull Throughput (MB): 11294.207336
Remote fast-pull Throughput (MB): 3369.501658
Local pull Throughput (MB): 3810.804663
Remote pull Throughput (MB): 2617.981305
Local push Throughput (MB): 4425.655896
Remote push Throughput (MB): 4405.089255

100k * 200

Local fast-pull Throughput (MB): 17946.883223
Remote fast-pull Throughput (MB): 5132.488473
Local pull Throughput (MB): 7042.887228
Remote pull Throughput (MB): 4292.009524
Local push Throughput (MB): 12931.302396
Remote push Throughput (MB): 5469.474048