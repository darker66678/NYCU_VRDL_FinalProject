# NYCU_VRDL_FinalProject
This is final project of VRDL, we challenged [RSNA Pneumonia Detection](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview)

| ```coco_trans.py``` generate coco json for [mmdetection](https://github.com/open-mmlab/mmdetection)

| ```yolo_label.py``` generate label ```.txt``` for [yolov5](https://github.com/ultralytics/yolov5)

| ```dcm_to_png.py``` transform ```.dcm``` to ```.png```

| ```requirements.txt``` all the dependencies

| ```data``` download [data](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) and put it in here

|```ensemble``` put the ensemble csv in here

|```mmdetection``` it's reference github [mmdetection](https://github.com/open-mmlab/mmdetection)

|```yolo_v5``` it's reference github [yolo_v5](https://github.com/ultralytics/yolov5)

|```ensemble.py``` and ```ensemble.sh``` ensemble model program

|```json_to_csv.py``` and ```json_to_csv_mmdet.py``` transform json to csv file

## Requirements
install all the dependencies ```pip install -r requirements.txt```

install pytorch ```sh install_pytorch.sh```

(we use CUDA 11.3)

Run ```sh ./mmdetection/install.sh``` (for Retinanet)
## Data preprocessing
First, download [data](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) and put it in ```data``` and run ```python dcm_to_png.py``` for preparing ```.png``` imgs

**if you want to use yolov5:**

Run ```python yolo_label.py``` for preparing yolo txt

Put ```train.txt```,```val.txt```,```test.txt``` in ```./yolo_v5/data/cfg```

**if you want to use mmdetection:**

Run ```python coco_trans.py``` for preparing coco json

Put 
## Training (Yolov5)
Run ```cd yolo_v5```

Run ```sh model.sh train``` quickly train

you can alter info of yolov5_train.sh

1.```--img```  img resolution 

2.```--batch``` batch size

3.```--epochs``` training epochs

4.```--data```  yaml path

5.```--weights``` version of yolov5

## Training (Retinanet)
Download [Retinanet_pretrained model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_mstrain_3x_coco/retinanet_x101_64x4d_fpn_mstrain_3x_coco_20210719_051838-022c2187.pth) and put it in ```./mmdetection/train_config/```

Run ```cd mmdetection```

Run ```sh model.sh train``` to train Retinanet model

## Inference(Ensemble model)

## Pre-trained model
Download [Retinanet_pretrained model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_mstrain_3x_coco/retinanet_x101_64x4d_fpn_mstrain_3x_coco_20210719_051838-022c2187.pth) and put it in ```./mmdetection/train_config/```
## Results


