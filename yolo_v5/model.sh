if [ $1 == "train" ]; then
    python train.py --img 512  --batch 48  --epochs 40  --data ./data/train.yaml --weights /home/tylin/vscode/VRDL/NYCU_VRDL_FinalProject/yolo_v5/runs/train/exp8/weights/40_epoch.pt
elif [ $1 == "test" ]; then
    python val.py --img 512 --batch 48 --save-json --data ./data/test.yaml --weights /home/tylin/vscode/VRDL/NYCU_VRDL_FinalProject/yolo_v5/runs/train/exp8/weights/40_epoch.pt --conf-thres 0.1 --iou-thres 0.1
    cd ..
    python json_to_csv.py --json ./yolo_v5/runs/val/exp/*.json
elif [ $1 == "ensemble" ]; then
    python val.py --weights ./runs/model/yolov5_448.pt ./runs/model/yolov5l6_512.pt ./runs/model/yolov5m6_512.pt ./runs/model/epoch_8.pt --data ./data/project.yaml --img 512 --batch 48 --save-json --conf-thres 0.1 --iou-thres 0.05 --augment
fi
