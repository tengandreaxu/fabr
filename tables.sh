#!/bin/bash
set -x;
set -e;
python3 paper/table_batch_and_nu.py;
python3 paper/table_cifar_expanded.py;
python3 paper/table_cifar_subsampled.py;