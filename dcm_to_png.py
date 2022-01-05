import pydicom
import matplotlib.pyplot as plt
import cv2
import os

if not os.path.isdir(r'./data/imgs/train/'):
    os.makedirs(r'./data/imgs/train/')

if not os.path.isdir(r'./data/imgs/test/'):
    os.makedirs(r'./data/imgs/test/')

path = r'./data/stage_2_train_images/'
train_list = os.listdir(path)
for img_name in train_list:
    ds = pydicom.read_file(path+img_name)
    img = ds.pixel_array
    out_path = r'./data/imgs/train/' + img_name[:-3] + "png"
    cv2.imwrite(out_path, img)

path = r'./data/stage_2_test_images/'
test_list = os.listdir(path)
for img_name in test_list:
    ds = pydicom.read_file(path+img_name)
    img = ds.pixel_array
    out_path = r'./data/imgs/test/' + img_name[:-3] + "png"
    cv2.imwrite(out_path, img)

print("finish!")
