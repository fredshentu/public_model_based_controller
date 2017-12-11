#!/bin/bash

initlrs=(1e-3 1e-4)
extratios=(0.9 0.99)
fwratios=(0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)

for initlr in ${initlrs[@]} ; do
    for extratio in ${extratios[@]}; do
        for fwratio in ${fwratios[@]} ; do
            python run_box3d_state_v4_tf_icm_sweep.py --init_lr $initlr --fw_ratio $fwratio --ext_ratio $extratio --tfboard_path /home/fred/tfboard_path
        done
    done
done