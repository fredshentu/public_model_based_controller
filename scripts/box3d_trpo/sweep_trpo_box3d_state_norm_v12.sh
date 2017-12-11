#!/bin/bash

initlrs=(1e-3 1e-4)
fwratios=(0.4 0.5 0.6 0.7 0.8 0.01 0.1 0.2 0.3)

for initlr in ${initlrs[@]} ; do
    for fwratio in ${fwratios[@]} ; do
    	python run_box3d_state_v12_tf_icm_sweep_norm.py --init_lr $initlr --fw_ratio $fwratio --tfboard_path /x/mujoco/tfboard_path
    done
done