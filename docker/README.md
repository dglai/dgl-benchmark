Build docker image for benchmark
================================

Files in this folder
--------------------
Dockerfile for benchmarking DGL models (with GPU support)
install folder contains scripts that install common dependencies

Build image
---------------
```bash
docker build -t dgl-benchmark -f Dockerfile .
```

Run container
--------------
```bash
nvidia-docker run -it --name benchmark dgl-benchmark
```
