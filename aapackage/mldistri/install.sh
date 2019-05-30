#!/bin/bash

reset

sudo apt-get update
sudo apt-get install -y wget python python-cffi python-pip
sudo pip install torchvision==0.2.0
sudo pip install torch==0.4.1 -f https://download.pytorch.org/whl/cpu/stable




mkdir openmpi
cd ./openmpi
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz
# steps from: https://www.open-mpi.org/faq/?category=building#easy-build
gunzip -c openmpi-4.0.1.tar.gz | tar xf -
cd openmpi-4.0.1
./configure --prefix=/usr/local
sudo make all install
sudo ldconfig




sudo pip install horovod

echo "Done!"
