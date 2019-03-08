#!/bin/bash

print_help() {
    echo "Usage : "
    echo "  --hsize=<hsize>        hidden size"
    echo "  --degree=<degree>      Out degrees for synthetic power law graph"
    echo "  --epochs=<epochs>      Epochs to train"
    echo ""
    exit 1
} # end of print_help

unknown_option() {
    echo "Unrecognized option : $1"
    print_help;
    exit 1
} # end of unknow option

#########################################
#
# Default parameters
#
########################################

hsize=16
degree=10
epochs=200

#########################################
#
# Parse command line arguments
#
########################################
while [ $# -gt 0 ]
do case $1 in
    --hsize=*)          hsize=${1##--hsize=} ;;
    --degree=*)         degree=${1##--degree=} ;;
    --epochs=*)         epochs=${1##--epochs=} ;;
    *) unknown_option $1 ;;
esac
shift
done

echo "Real datasets"
echo "Time to train 200 epochs with hidden size ${hsize}:"

echo "Cora:"
time_elapsed=`python3 gcn_spmv.py --dataset cora        \
                                  --n-hidden ${hsize}   \
                                  --n-epochs ${epochs}  \
                                  --gpu 0               \
                                  2>/dev/null | tail -n 1`
echo "${time_elapsed} seconds"
echo "CiteSeer:"
time_elapsed=`python3 gcn_spmv.py --dataset citeseer    \
                                  --n-hidden ${hsize}   \
                                  --n-epochs ${epochs}  \
                                  --gpu 0               \
                                  2>/dev/null | tail -n 1`
echo "${time_elapsed} seconds"
echo "PubMed:"
time_elapsed=`python3 gcn_spmv.py --dataset pubmed      \
                                  --n-hidden ${hsize}   \
                                  --n-epochs ${epochs}  \
                                  --gpu 0               \
                                  2>/dev/null | tail -n 1`
echo "${time_elapsed} seconds"

echo

echo "Synthetic graphs with in degree ${degree} and power-law out degree"
echo "Time to train 200 epochs with hidden size ${hsize}:"

nodes=10000
echo "Number of nodes: ${nodes}"
time_elapsed=`python3 gcn_spmv.py --dataset synthetic   \
                                  --n-hidden ${hsize}   \
                                  --n-epochs ${epochs}  \
                                  --n-nodes ${nodes}    \
                                  --degree ${degree}    \
                                  --gpu 0               \
                                  2>/dev/null | tail -n 1`
echo "${time_elapsed} seconds"

nodes=50000
echo "Number of nodes: ${nodes}"
time_elapsed=`python3 gcn_spmv.py --dataset synthetic   \
                                  --n-hidden ${hsize}   \
                                  --n-epochs ${epochs}  \
                                  --n-nodes ${nodes}    \
                                  --degree ${degree}    \
                                  --gpu 0               \
                                  2>/dev/null | tail -n 1`
echo "${time_elapsed} seconds"

nodes=100000
echo "Number of nodes: ${nodes}"
time_elapsed=`python3 gcn_spmv.py --dataset synthetic   \
                                  --n-hidden ${hsize}   \
                                  --n-epochs ${epochs}  \
                                  --n-nodes ${nodes}    \
                                  --degree ${degree}    \
                                  --gpu 0               \
                                  2>/dev/null | tail -n 1`
echo "${time_elapsed} seconds"
