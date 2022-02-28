#! /bin/bash --


epochs=50
levels=5
lr=0.1
for seed in 1 2 3
do
    for sparsity in 0.01 0.05 0.1 0.2 0.5
    do
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python3 cifar_main.py --sparsity $sparsity --anneal True --initBias kn-nonzero-bias --epochs $epochs --levels $levels --lr $lr --plant-model | tee log_kn_zerobias_seed${seed}_sparsity${sparsity}.txt
    done
done 
