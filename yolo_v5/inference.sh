#!/bin/bash

echo "functions : ${1}"

model="yolov5m.yaml"
pretrained="yolov5m6.pt"
val_task="test"

running_id=${2}

dataset_0="data/final_0.yaml"
dataset_1="data/final_1.yaml"
dataset_2="data/final_2.yaml"
dataset_3="data/final_3.yaml"
dataset_4="data/final_4.yaml"
img_size=320
img_size_0=640
img_size_1=576
img_size_2=512
img_size_3=448
img_size_4=384
batch_size_0=640
batch_size_1=576
batch_size_2=512
batch_size_3=448
batch_size_4=384

epoch=50

conf_thres=0.01
iou_thres=0.1

weights_1="best_5-1.pt"
weights_2="best_5-2.pt"
weights_3="best_5-3.pt"
weights_4="best_5-4.pt"
weights_5="best_5-5.pt"

# python sharpen.py --dir test --target test_sharp
# python formating.py -train images/train_sharp -test images/test_sharp -lpf labels/train_sharp

if [ ${1} = "train" ]; then
    echo "in training mode"
    python3 train.py --img $img_size_0 --batch-size $batch_size_0 --epochs $epoch --data $dataset_0 --cfg $model --weights $pretrained --project "runs/$running_id"
    python3 train.py --img $img_size_1 --batch-size $batch_size_1 --epochs $epoch --data $dataset_1 --cfg $model --weights $pretrained --project "runs/$running_id"
    python3 train.py --img $img_size_2 --batch-size $batch_size_2 --epochs $epoch --data $dataset_2 --cfg $model --weights $pretrained --project "runs/$running_id"
    python3 train.py --img $img_size_3 --batch-size $batch_size_3 --epochs $epoch --data $dataset_3 --cfg $model --weights $pretrained --project "runs/$running_id"
    python3 train.py --img $img_size_4 --batch-size $batch_size_4 --epochs $epoch --data $dataset_4 --cfg $model --weights $pretrained --project "runs/$running_id"

elif [ ${1} = "test" ]; then
    echo "in testing mode"
    # 1
    echo "setting 1"
    python3 val.py --img $img_size_0 --data $dataset_0 --task $val_task --weights $weights_1 --conf-thres 0.01 --iou-thres $iou_thres --project "runs/$running_id" --save-json 
    mv answer.json answer_1.json
    python3 json_to_csv.py --json answer_1.json
    # 2
    echo "setting 2"
    python3 val.py --img $img_size_0 --data $dataset_1 --task $val_task --weights $weights_2 --conf-thres 0.01 --iou-thres $iou_thres --project "runs/$running_id" --save-json 
    mv answer.json answer_2.json
    python3 json_to_csv.py --json answer_2.json
    # 3
    echo "setting 3"
    python3 val.py --img $img_size_0 --data $dataset_2 --task $val_task --weights $weights_3 --conf-thres 0.01 --iou-thres $iou_thres --project "runs/$running_id" --save-json 
    mv answer.json answer_3.json
    python3 json_to_csv.py --json answer_3.json
    # 4
    echo "setting 4"
    python3 val.py --img $img_size_0 --data $dataset_3 --task $val_task --weights $weights_4 --conf-thres 0.01 --iou-thres $iou_thres --project "runs/$running_id" --save-json 
    mv answer.json answer_4.json
    python3 json_to_csv.py --json answer_4.json
    # 5
    echo "setting 5"
    python3 val.py --img $img_size_0 --data $dataset_4 --task $val_task --weights $weights_5 --conf-thres 0.01 --iou-thres $iou_thres --project "runs/$running_id" --save-json 
    mv answer.json answer_5.json
    python3 json_to_csv.py --json answer_5.json
    # 6
    echo "setting 6"
    python3 val.py --img $img_size_2 --data $dataset_0 --task $val_task --weights $weights_1 --conf-thres 0.01 --iou-thres $iou_thres --project "runs/$running_id" --save-json 
    mv answer.json answer_6.json
    python3 json_to_csv.py --json answer_6.json
    # 7
    echo "setting 7"
    python3 val.py --img $img_size_2 --data $dataset_1 --task $val_task --weights $weights_2 --conf-thres 0.01 --iou-thres $iou_thres --project "runs/$running_id" --save-json 
    mv answer.json answer_7.json
    python3 json_to_csv.py --json answer_7.json
    # 8
    echo "setting 8"
    python3 val.py --img $img_size_2 --data $dataset_2 --task $val_task --weights $weights_3 --conf-thres 0.01 --iou-thres $iou_thres --project "runs/$running_id" --save-json 
    mv answer.json answer_8.json
    python3 json_to_csv.py --json answer_8.json
    # 9
    echo "setting 9"
    python3 val.py --img $img_size_2 --data $dataset_3 --task $val_task --weights $weights_4 --conf-thres 0.01 --iou-thres $iou_thres --project "runs/$running_id" --save-json 
    mv answer.json answer_9.json
    python3 json_to_csv.py --json answer_9.json
    # 10
    echo "setting 10"
    python3 val.py --img $img_size_2 --data $dataset_4 --task $val_task --weights $weights_5 --conf-thres 0.01 --iou-thres $iou_thres --project "runs/$running_id" --save-json 
    mv answer.json answer_10.json
    python3 json_to_csv.py --json answer_10.json
    # 11
    echo "setting 11"
    python3 val.py --img $img_size --data $dataset_0 --task $val_task --weights $weights_1 --conf-thres 0.001 --iou-thres $iou_thres --project "runs/$running_id" --save-json 
    mv answer.json answer_11.json
    python3 json_to_csv.py --json answer_11.json
    # 12
    echo "setting 12"
    python3 val.py --img $img_size --data $dataset_1 --task $val_task --weights $weights_2 --conf-thres 0.001 --iou-thres $iou_thres --project "runs/$running_id" --save-json 
    mv answer.json answer_12.json
    python3 json_to_csv.py --json answer_12.json
    # 13
    echo "setting 13"
    python3 val.py --img $img_size --data $dataset_2 --task $val_task --weights $weights_3 --conf-thres 0.001 --iou-thres $iou_thres --project "runs/$running_id" --save-json 
    mv answer.json answer_13.json
    python3 json_to_csv.py --json answer_13.json
    # 14
    echo "setting 14"
    python3 val.py --img $img_size --data $dataset_3 --task $val_task --weights $weights_4 --conf-thres 0.001 --iou-thres $iou_thres --project "runs/$running_id" --save-json 
    mv answer.json answer_14.json
    python3 json_to_csv.py --json answer_14.json
    # 15
    echo "setting 15"
    python3 val.py --img $img_size --data $dataset_4 --task $val_task --weights $weights_5 --conf-thres 0.001 --iou-thres $iou_thres --project "runs/$running_id" --save-json 
    mv answer.json answer_15.json
    python3 json_to_csv.py --json answer_15.json

    # ensemble
    python ensemble.py --conf_thr 0.12 --ens_thr 0.5 --iou_thr 0.001 --folder ./
    zip ensemble.zip ensemble.csv


else
    echo "args error : you should specify an arg (train/test/det)"
fi