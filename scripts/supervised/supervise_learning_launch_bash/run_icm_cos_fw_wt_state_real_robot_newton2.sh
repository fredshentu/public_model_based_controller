gpu=$1
fweight=$2
initlr=$3
name=$(echo icm_supervised_real_robot_v11_fw_${fweight}_lr_${initlr})
CUDA_VISIBLE_DEVICES=$gpu python train_icm_supervised_wt_state.py real_robot /media/icm_data/tf_record --init_lr $initlr  --cos_forward --forward_weight $fweight --tfmodel_path "/media/4tb/fred/tfmodel_real_robot/$name.pkl" --tfboard_path "/media/4tb/fred/tfboard_real_robot/icm_supervise/$name"