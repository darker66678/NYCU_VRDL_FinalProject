if [ $1 == "train" ]; then
    python tools/train.py ./train_config/retinanet_x101_64x4d_fpn_2x_coco.py
elif [ $1 == "test" ]; then
    python tools/test.py  $2  $3 --format-only --options "jsonfile_prefix=./$4" 
elif [ $1 == "show" ]; then
    python tools/test.py /home/tylin/vscode/VRDL/NYCU_VRDL_FinalProject/mmdetection/train_config/vfnet_r101_fpn_mstrain_2x_coco/posi_3/vfnet_r101_fpn_mstrain_2x_coco.py $2    --show
fi

