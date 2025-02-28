#!/bin/bash
# conda activate hgnn

python train.py --name SAGE_l5 --dataset roman-empire --model SAGE --num_layers 5 --device cuda:0 --verbose 

python train.py --name SAGE_l2 --dataset amazon-ratings --model SAGE --num_layers 2 --device cuda:0  --verbose 


python train.py --name SAGE_l5 --dataset minesweeper --model SAGE --num_layers 5 --device cuda:0 --verbose 

python train.py --name SAGE_l5 --dataset tolokers --model SAGE --num_layers 5 --device cuda:0 --verbose 

python train.py --name SAGE_l5 --dataset questions --model SAGE --num_layers 5 --device cuda:0 --verbose 
