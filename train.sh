#! /bin/bash

python main.py --arch resnet --train_root ./data/DUTS/DUTS-TR --train_list ./data/DUTS/DUTS-TR/train_pair.lst
# you can optionly change the -lr and -wd params
