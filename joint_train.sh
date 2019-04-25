#! /bin/bash

python joint_main.py --arch resnet --train_root ./data/DUTS/DUTS-TR --train_list ./data/DUTS/DUTS-TR/train_pair.lst --train_edge_root ./data/HED-BSDS_PASCAL --train_edge_list ./data/HED-BSDS_PASCAL/bsds_pascal_train_pair_r_val_r_small.lst
# you can optionly change the -lr and -wd params
