import os
import pandas as pd
import random


def convert(size, box):
    dw = size[0]
    dh = size[1]
    x = box[0] + (box[2]/2.0)
    y = box[1] + (box[3]/2.0)
    w = box[2]
    h = box[3]
    x = x/dw
    w = w/dw
    y = y/dh
    h = h/dh
    return (x, y, w, h)


patientId = None

out_path = r'./data/imgs/train/'
data = pd.read_csv(r'./data/stage_2_train_labels.csv')
for i in range(len(data)):
    target = data['Target'].iloc[i]

    if(target):
        if data['patientId'].iloc[i] != patientId:
            patientId = data['patientId'].iloc[i]
            txt_outfile = open(out_path+patientId+'.txt', "w")
        x = data['x'].iloc[i]
        y = data['y'].iloc[i]
        w = data['width'].iloc[i]
        h = data['height'].iloc[i]
        yolo_x, yolo_y, yolo_w, yolo_h = convert([1024, 1024], [x, y, w, h])
        bbox = [float(yolo_x), float(yolo_y), float(yolo_w), float(yolo_h)]
        txt_outfile.write(
            str(0) + " " + " ".join([str(i) for i in bbox]) + '\n')


total_1 = 0
total_0 = 0
patientId = None
lines = []

for i in range(len(data)):
    if data['patientId'].iloc[i] != patientId:
        patientId = data['patientId'].iloc[i]
        # relative to the path you executive train.py of yolov5
        lines.append('../data/imgs/train/'+patientId+'.png' + '\n')
        #txt_outfile.write('../data/imgs/train/'+patientId+'.png' + '\n')

train_num = int(len(lines)*0.95)
# training
txt_outfile = open('./data/train.txt', "w")
for line in lines[:train_num]:
    txt_outfile.write(line)

# validation
txt_outfile = open('./data/val.txt', "w")
for line in lines[train_num:]:
    txt_outfile.write(line)

# testing
data = pd.read_csv(r'./data/stage_2_sample_submission.csv')
txt_outfile = open('./data/test.txt', "w")
patientId = None
for i in range(len(data)):
    if data['patientId'].iloc[i] != patientId:
        patientId = data['patientId'].iloc[i]
        # relative to the path you executive train.py of yolov5
        txt_outfile.write('../data/imgs/test/'+patientId+'.png' + '\n')

print("finish!")

'''# 1-data
for i in range(len(data)):
    if data['patientId'].iloc[i] != patientId and data['Target'].iloc[i] == 1:
        patientId = data['patientId'].iloc[i]
        total_1 = total_1+1
        # relative to the path you executive train.py of yolov5
        lines.append('../data/imgs/train/'+patientId+'.png' + '\n')
        #txt_outfile.write('../data/imgs/train/'+patientId+'.png' + '\n')'''

'''
# 0-data
for i in range(len(data)):
    if data['patientId'].iloc[i] != patientId and data['Target'].iloc[i] == 0:
        patientId = data['patientId'].iloc[i]
        total_0 = total_0+1
        lines.append('../data/imgs/train/'+patientId+'.png' + '\n')
        #txt_outfile.write('../data/imgs/train/'+patientId+'.png' + '\n')
        if(total_0 == total_1):
            break'''

'''
txt_outfile = open('./data/train.txt', "w")
print("posivitve data :", len(lines))
for line in lines[:train_num]:
    txt_outfile.write(line)
txt_outfile = open('./data/val.txt', "w")
for line in lines[train_num:]:
    txt_outfile.write(line)'''
