1. change `ip_config.txt` to your instance IP.

2. Start server:

In instance_0:

`DGLBACKEND=pytorch python3 server.py --num_client 2 --server_id 0 &`

In instance_1:

`DGLBACKEND=pytorch python3 server.py --num_client 2 --server_id 1 &`

3. Start client:

In instance_0:

`DGLBACKEND=pytorch python3 client.py`

In instance_1:

`DGLBACKEND=pytorch python3 client.py`