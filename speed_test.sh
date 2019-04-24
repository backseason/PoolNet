#! /bin/bash

# 1 gpu id ,2 python file name, # 3 results folder name

ARRAY=(m_r)
ELEMENTS=${#ARRAY[@]}

echo "Testing on GPU " $1 " with file " $2 " to " $3

for (( i=0;i<$ELEMENTS;i++)); do
    CUDA_VISIBLE_DEVICES=$1 python $2 --mode='test' --model=$3'/models/final.pth' --test_fold=$3'-sal-'${ARRAY[${i}]} --sal_mode=${ARRAY[${i}]}
done

echo "Speed test done."
