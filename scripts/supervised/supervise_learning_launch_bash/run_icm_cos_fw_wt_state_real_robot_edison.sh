gpu=$1
fweight=$2
name="icm_supervised_real_robot_v11_fw_$fweight"
CUDA_VISIBLE_DEVICES=$gpu python train_icm_supervised_wt_state.py real_robot /home/fredshentu/Desktop/real_robot/random_data/episode_data/tf_record --cos_forward --forward_weight $fweight --tfmodel_path "/home/dianchen/tfmodel_real_robot/$name.pkl" --tfboard_path "/home/dianchen/tfboard_real_robot/$name"