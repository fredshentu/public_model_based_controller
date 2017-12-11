#!/usr/bin/env bash

policy_path=$1
for i in {0..800..200}
do 
	python sim_policy.py $policy_path/itr_$i.pkl --icm --analyze_forward --data_path /home/fred/forward_state_data/one_box/itr_$i
done