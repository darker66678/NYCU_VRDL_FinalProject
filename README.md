# NYCU_VRDL_FinalProject
This is final project of VRDL, we challenged [RSNA Pneumonia Detection](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview)

| ```coco_trans.py``` generate coco json for [mmdetection](https://github.com/open-mmlab/mmdetection)

| ```yolo_label.py``` generate label ```.txt``` for [yolov5](https://github.com/ultralytics/yolov5)

| ```dcm_to_png.py``` transform ```.dcm``` to ```.png```

| ```requirements.txt``` all the dependencies

| ```data``` download [data](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) and put it in here
## installation
install all the dependencies ```pip install -r requirements.txt```
## data preparation
First, run ```python dcm_to_png.py``` for preparing ```.png``` imgs

**if you want to use yolov5:**

Run ```python yolo_label.py``` for preparing yolo txt

**if you want to use mmdetection:**

Run ```python coco_trans.py``` for preparing coco json