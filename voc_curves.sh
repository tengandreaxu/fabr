#!/bin/bash
set -x;
set -e;
python3 plotting/voc_curves_batch_nu.py
python3 plotting/voc_curves.py