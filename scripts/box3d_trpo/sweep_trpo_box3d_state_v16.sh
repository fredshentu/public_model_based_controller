#!/bin/bash

initlrs=(1e-3 1e-4)
fwratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.01)

for initlr in ${initlrs[@]} ; do
    for fwratio in ${fwratios[@]} ; do
    	python run_box3d_state_v16_tf_icm_sweep.py --init_lr $initlr --fw_ratio $fwratio --tfboard_path /x/mujoco/tfboard_path
    done
done