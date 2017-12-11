gpu=$1
fweight=$2
initlr=$3
name=$(echo icm_supervised_box3dpixel_v11_new_cnn_fw_cos_${fweight}_lr_${initlr})
CUDA_VISIBLE_DEVICES=$gpu python train_icm_supervised.py Box3dReachPixel-v11 /media/icm_data/v11-box-dense-2e3 --cos_forward --init_lr $initlr --forward_weight $fweight --tfmodel_path "/media/4tb/fred/tfmodel_box3d/$name.pkl" --tfboard_path "/media/4tb/fred/tfboard_box3d/$name"
