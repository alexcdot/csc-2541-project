#!/bin/bash

seed=1
while [ $seed -le 2 ]
do
  python3 train_el2n.py -c configs/el2n_cifar10_res18.json --seed $seed
  ((seed++))
done

python3 train_el2n.py -c configs/el2n_subsample_cifar10_res18.json