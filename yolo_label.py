import os
import pandas as pd


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


out_path = r'./data/labels/train/'
data = pd.read_csv(r'./data/stage_2_train_labels.csv')
for i in range(len(data)):
    target = data['Target'].iloc[i]
    if(target):
        patientId = data['patientId'].iloc[i]
        x = data['x'].iloc[i]
        y = data['y'].iloc[i]
        w = data['width'].iloc[i]
        h = data['height'].iloc[i]
        yolo_x, yolo_y, yolo_w, yolo_h = convert([1024, 1024], [x, y, w, h])
        bbox = [float(yolo_x), float(yolo_y), float(yolo_w), float(yolo_h)]
        txt_outfile = open(out_path+patientId+'.txt', "w")
        txt_outfile.write(
            str(0) + " " + " ".join([str(i) for i in bbox]) + '\n')

txt_outfile = open('./data/train.txt', "w")
for i in range(len(data)):
    patientId = data['patientId'].iloc[i]
    txt_outfile.write( '../train/'+patientId+'.png' +'\n')

print("finish!")
