#!/bin/bash

reset
which conda


echo "Conda Install"
conda env create -n py36tch -f py36tch.yml


echo "Open MPI"

mkdir openmpi
cd ./openmpi
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz
# steps from: https://www.open-mpi.org/faq/?category=building#easy-build
gunzip -c openmpi-4.0.1.tar.gz | tar xf -
cd openmpi-4.0.1
./configure --prefix=/usr/local
sudo make all install
sudo ldconfig


echo "Done!"


