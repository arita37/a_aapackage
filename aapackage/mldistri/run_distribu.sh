#!/bin/sh

mpirun -np 4 \
    -H localhost:4 \
    --display-map \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python pytorch_mnist_horovod_cpu.py
    
    
    
    