#!/bin/bash
# conda activate hgnn

python train.py --name SAGE_l1 --dataset roman-empire --model SAGE --num_layers 1 --device cuda:0
python train.py --name SAGE_l2 --dataset roman-empire --model SAGE --num_layers 2 --device cuda:0
python train.py --name SAGE_l3 --dataset roman-empire --model SAGE --num_layers 3 --device cuda:0
python train.py --name SAGE_l4 --dataset roman-empire --model SAGE --num_layers 4 --device cuda:0
python train.py --name SAGE_l5 --dataset roman-empire --model SAGE --num_layers 5 --device cuda:0 --verbose 

python train.py --name SAGE_l1 --dataset amazon-ratings --model SAGE --num_layers 1 --device cuda:0
python train.py --name SAGE_l2 --dataset amazon-ratings --model SAGE --num_layers 2 --device cuda:0
python train.py --name SAGE_l3 --dataset amazon-ratings --model SAGE --num_layers 3 --device cuda:0 --verbose
python train.py --name SAGE_l4 --dataset amazon-ratings --model SAGE --num_layers 4 --device cuda:0
python train.py --name SAGE_l5 --dataset amazon-ratings --model SAGE --num_layers 5 --device cuda:0

python train.py --name SAGE_l1 --dataset minesweeper --model SAGE --num_layers 1 --device cuda:0
python train.py --name SAGE_l2 --dataset minesweeper --model SAGE --num_layers 2 --device cuda:0
python train.py --name SAGE_l3 --dataset minesweeper --model SAGE --num_layers 3 --device cuda:0
python train.py --name SAGE_l4 --dataset minesweeper --model SAGE --num_layers 4 --device cuda:0
python train.py --name SAGE_l5 --dataset minesweeper --model SAGE --num_layers 5 --device cuda:0

python train.py --name SAGE_l1 --dataset tolokers --model SAGE --num_layers 1 --device cuda:0
python train.py --name SAGE_l2 --dataset tolokers --model SAGE --num_layers 2 --device cuda:0
python train.py --name SAGE_l3 --dataset tolokers --model SAGE --num_layers 3 --device cuda:0
python train.py --name SAGE_l4 --dataset tolokers --model SAGE --num_layers 4 --device cuda:0
python train.py --name SAGE_l5 --dataset tolokers --model SAGE --num_layers 5 --device cuda:0

python train.py --name SAGE_l1 --dataset questions --model SAGE --num_layers 1 --device cuda:0
python train.py --name SAGE_l2 --dataset questions --model SAGE --num_layers 2 --device cuda:0
python train.py --name SAGE_l3 --dataset questions --model SAGE --num_layers 3 --device cuda:0
python train.py --name SAGE_l4 --dataset questions --model SAGE --num_layers 4 --device cuda:0
python train.py --name SAGE_l5 --dataset questions --model SAGE --num_layers 5 --device cuda:0
