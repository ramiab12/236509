#!/bin/bash
conda env update -f environment.yml 
conda activate cuda_env
cd cuda
pip3 install -e .
python parallel_matmul.py
