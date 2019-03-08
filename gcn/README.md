Graph Convolutional Networks (GCN)
============

Run one dataset
----------------
Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python train.py --dataset cora --gpu 0
```

Run benchmark with real and synthetic datasets
-----------------------------------------------
```bash
bash benchmark.sh --hsize=16 --degree=10 --epochs=200
```
