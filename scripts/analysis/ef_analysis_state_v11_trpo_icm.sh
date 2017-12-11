#!/bin/bash

datafile=/home/dianchen/rllab/data/local/trpo-box3d-state-v10-tf-icm-cos-new/trpo_box3d_state_v10_tf_icm_cos_new_2017_06_27_03_31_02_0001/
filenames=($(ls ${datafile} | grep pkl))
for filename in ${filenames[@]} ; do
	python /home/dianchen/rail-rl/sim_policy.py $datafile$filename --icm --analyze_ef_range --data_path /home/dianchen/corl/data/ef_xyz_analysis/trpo_icm_state_v10_large/$filename
done