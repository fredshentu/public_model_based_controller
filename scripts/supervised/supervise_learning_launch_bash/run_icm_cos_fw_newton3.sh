gpu=$1
fweight=$2
initlr=$3
name=$(echo icm_supervised_box3dpixel_v11_fw_${fweight}_lr_${initlr})
CUDA_VISIBLE_DEVICES=$gpu python train_icm_supervised.py Box3dReachPixel-v11 /z/dianchen/v11 --cos_forward --init_lr $initlr --forward_weight $fweight --tfmodel_path "/z/dianchen/tfmodel_box3d/$name.pkl" --tfboard_path "/z/dianchen/tfboard_box3d/$name"