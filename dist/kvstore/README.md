1. Go to `pytorch` or `mxnet` folder and change `ip_config.txt` to your instance IP.

2. `rm *-shape` remove temp file

3. Start server:

In instance_0:

`DGLBACKEND=pytorch python3 server.py --num_client 2 --server_id 0 &`

In instance_1:

`DGLBACKEND=pytorch python3 server.py --num_client 2 --server_id 1 &`

4. Start client:

In instance_0:

`DGLBACKEND=pytorch python3 client.py`

In instance_1:

`DGLBACKEND=pytorch python3 client.py`