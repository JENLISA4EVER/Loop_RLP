#!/bin/bash

source ~/.bashrc
source ~/miniconda3/bin/activate
conda activate ouro

accelerate launch omni_math_entropy_split.py 