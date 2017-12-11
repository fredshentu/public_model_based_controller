#!/bin/bash

gpu=$1
gpuratio=$2
seed=$3

qfdecays=(0.0 1e-2)
qftaus=(1e-2 1e-3)
outhetas=(0.15 0.35 0.55)
ousigmas=(0.05 0.2 0.4)

for qfdecay in ${qfdecays[@]} ; do
    for qftau in ${qftaus[@]} ; do
        for outheta in ${outhetas[@]} ; do
            for ousigma in ${ousigmas[@]}; do
                CUDA_VISIBLE_DEVICES=$gpu python train_ddpg_state_v11_tf.py \
                --seed $seed \
                --qf_decay $qfdecay \
                --qf_soft_tau $qftau \
                --ou_theta $outheta \
                --ou_sigma $ousigma \
                --gpu_ratio $gpuratio \
                --tfboard_path /home/fshentu/box3d_ddpg
            done
        done
    done
done