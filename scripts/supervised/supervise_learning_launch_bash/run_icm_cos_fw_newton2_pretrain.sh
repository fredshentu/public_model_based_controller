gpu=$1
fweight=$2
initlr=$3
name=$(echo icm_supervised_box3dpixel_v11_new_fw_${fweight}_lr_${initlr}_pretrained)
CUDA_VISIBLE_DEVICES=$gpu python train_icm_supervised.py Box3dReachPixel-v11 /media/4tb/fred/v11-dian --restore --cos_forward --init_lr $initlr --forward_weight $fweight --tfmodel_path "/media/4tb/fred/tfmodel_box3d/$name.pkl" --tfboard_path "/media/4tb/fred/tfboard_box3d/$name"
