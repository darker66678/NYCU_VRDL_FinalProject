if [ $1 == "train" ]; then
    python train.py --img 512  --batch 48  --epochs 40  --data ./data/train.yaml --weights yolov5m6.pt
elif [ $1 == "test" ]; then
    python val.py --img 512 --batch 48 --save-json --data ./data/test.yaml --weights $2 --conf-thres 0.1 --iou-thres 0.1
fi
