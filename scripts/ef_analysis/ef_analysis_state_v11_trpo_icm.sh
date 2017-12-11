#!/bin/bash

datafile=/home/dianchen/rllab/data/local/trpo-state-v12-tf-icm-fw0.1-initlr-0.001/trpo-state-v12-tf-icm-fw0.1-initlr-0.001_2017_07_16_22_12_20_0001/
filenames=($(ls ${datafile} | grep pkl))
for filename in ${filenames[@]} ; do
	python /home/dianchen/rail-rl/sim_policy.py $datafile$filename --icm --analyze_ef_range --data_path /home/dianchen/corl/data/ef_xyz_analysis/trpo_icm_state_v12/$filename
done