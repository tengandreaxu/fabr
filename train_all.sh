#!/bin/bash

set -x;
set -e;

# CIFAR-10

#****************
# Subsample
# n \in {10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480}
#****************
python3 experiments/train_fabr_subsample.py;
python3 experiments/train_fabr_batch_subsample.py;
python3 experiments/train_fabr_nu_subsample.py;


#****************
# Full Sample 
# n = 50000
#****************
./scripts/train_fabr_on_cifar.sh;

# ResNet-34
python3 benchmarks/resnet_cifar_expanded.py;