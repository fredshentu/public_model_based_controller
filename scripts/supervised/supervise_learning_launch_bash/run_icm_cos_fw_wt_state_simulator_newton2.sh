gpu=$1
fweight=$2
initlr=$3
name=$(echo icm_supervised_v11_wt_state_cos${fweight}_lr_${initlr})
CUDA_VISIBLE_DEVICES=$gpu python train_icm_supervised_wt_state.py Box3dReachPixel-v11 /media/icm_data/v11-wt-armstate --init_lr $initlr  --cos_forward --forward_weight $fweight --tfmodel_path "/media/4tb/fred/tfmodel_box3d/pixel_state/$name.pkl" --tfboard_path "/media/4tb/fred/tfboard_box3d/pixel_state/$name"