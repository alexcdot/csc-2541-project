#!/bin/bash

python3 ./train_hyper_tune.py -c configs/el2n_cifar100_res18_precomputed_epoch1.json
python3 ./train_hyper_tune.py -c configs/el2n_cifar100_res18_precomputed_epoch10.json
python3 ./train_hyper_tune.py -c configs/el2n_cifar100_res18_precomputed_epoch100.json
