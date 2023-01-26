#!/bin/bash

set -e;
set -x;



# python3 experiments/train_fabr.py \
#     --dataset cifar10 \
#     --model-name FABR \
#     --max-multiplier 15 \
#     --runs 5 \
#     --shift-random-seed \
#     --voc-curve \
#     --reduce-voc-curve 50000 \
#     --p1 20000 \
#     --global-average-pool \
#     --sample 0;

python3 experiments/train_fabr.py \
    --dataset cifar10 \
    --model-name FABRBatch \
    --max-multiplier 15 \
    --shift-random-seed \
    --runs 5 \
    --p1 50000 \
    --voc-curve \
    --batch-no-convolution \
    --reduce-voc-curve 50000 \
    --global-average-pool \
    --batch-size 2000 \
    --pred-batch-size 5000 \
    --sample 0;

# python3 experiments/train_fabr.py \
#     --dataset cifar10 \
#     --model-name FABRNu \
#     --shift-random-seed \
#     --max-multiplier 15 \
#     --runs 5 \
#     --p1 8000 \
#     --reduce-voc-curve 50000 \
#     --voc-curve \
#     --global-average-pool \
#     --nu 2000 \
#     --sample 0;
