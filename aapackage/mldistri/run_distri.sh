#!/bin/sh

echo "Distributed Pytorch HVD Starting"

if [ "$1" ]; then
    n_node="$1"
else
n_node=2
fi



mpirun -np $n_node \
    -H localhost:$n_node \
    --display-map \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python tch_distribu.py
    
    
    
    