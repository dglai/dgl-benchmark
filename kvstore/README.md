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

10k * 10

Local fast-pull Throughput (MB): 649.258425
Remote fast-pull Throughput (MB): 173.023789
Local pull Throughput (MB): 291.773802
Remote pull Throughput (MB): 162.708141
Local push Throughput (MB): 442.425876
Remote push Throughput (MB): 479.007257

100k * 10

Local fast-pull Throughput (MB): 1314.218752
Remote fast-pull Throughput (MB): 574.816740
Local pull Throughput (MB): 552.659689
Remote pull Throughput (MB): 286.657311
Local push Throughput (MB): 1045.716321
Remote push Throughput (MB): 907.782533

100k * 200

Local fast-pull Throughput (MB): 6666.765678
Remote fast-pull Throughput (MB): 581.152885
Local pull Throughput (MB): 2375.640861
Remote pull Throughput (MB): 689.765973
Local push Throughput (MB): 4990.976247
Remote push Throughput (MB): 1039.765263


———————————

4 client & 4 server

10k * 10

Local fast-pull Throughput (MB): 2751.869516
Local pull Throughput (MB): 1314.717302
Remote pull Throughput (MB): 625.735320
Local push Throughput (MB): 1918.196895
Remote push Throughput (MB): 1637.778081

100k * 10

Local fast-pull Throughput (MB): 4519.840511
Remote fast-pull Throughput (MB): 1407.205000
Local pull Throughput (MB): 1404.302542
Remote pull Throughput (MB): 987.953401
Local push Throughput (MB): 2445.326418
Remote push Throughput (MB): 2779.199142

100k * 200

Local fast-pull Throughput (MB): 12242.980742
Remote fast-pull Throughput (MB): 1815.319213
Local pull Throughput (MB): 4707.276956
Remote pull Throughput (MB): 1902.349101
Local push Throughput (MB): 9012.186808
Remote push Throughput (MB): 3421.485186

—————————————

20 client & 20 server

10k * 10

Local fast-pull Throughput (MB): 7244.667906
Remote fast-pull Throughput (MB): 1039.229844
Local pull Throughput (MB): 3239.166927
Remote pull Throughput (MB): 1307.803466
Local push Throughput (MB): 4157.319914
Remote push Throughput (MB): 2852.975899

100k * 10

Local fast-pull Throughput (MB): 5839.269544
Remote fast-pull Throughput (MB): 1654.370116
Local pull Throughput (MB): 2224.554786
Remote pull Throughput (MB): 1500.352231
Local push Throughput (MB): 2679.853032
Remote push Throughput (MB): 2811.585548

100k * 200

Local fast-pull Throughput (MB): 17946.883223
Remote fast-pull Throughput (MB): 5132.488473
Local pull Throughput (MB): 7042.887228
Remote pull Throughput (MB): 4292.009524
Local push Throughput (MB): 12931.302396
Remote push Throughput (MB): 5469.474048