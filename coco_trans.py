import pandas as pd
import json
import os
from tqdm import tqdm
from pycocotools import mask as cocomask

category_ids = {
    "Lung_Opacity": 1,
}

def images_annotations_info(image_id, annotations, data_info):
    for i in range(len(data_info)):
        if(data_info['Target'].iloc[i]):
            x = data_info['x'].iloc[i]
            y = data_info['y'].iloc[i]
            w = data_info['width'].iloc[i]
            h = data_info['height'].iloc[i]
            annotation = {
                "area": w*h,
                "iscrowd": 0,
                "image_id": image_id,
                'bbox': [x, y, w, h],
                "category_id": category_id,
                "id": len(annotations)
            }
            annotations.append(annotation)

    return annotations


def get_coco_json_format():
    # Standard COCO format
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format


def create_category_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key
        }
        category_list.append(category)

    return category_list


def create_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id
    }

    return images


if __name__ == "__main__":
    # Get the standard COCO JSON format
    path = './data/imgs/train/'
    csv_path = './data/stage_2_train_labels.csv'
    allFileList = os.listdir(path)
    data = pd.read_csv(r'./data/stage_2_train_labels.csv')
    print("preparing train.json")
    image_id = 1
    coco_format = get_coco_json_format()
    coco_format["categories"] = create_category_annotation(category_ids)
    coco_format["images"] = []
    coco_format["annotations"] = []
    w = 1024
    h = 1024
    category_id = 1

    for file in tqdm(allFileList):
        img_info = create_image_annotation(
            file, w, h, image_id)
        coco_format["images"].append(img_info)

        img_name = file[:-4]
        data_info = data[data['patientId'] == img_name]
        coco_format["annotations"] = images_annotations_info(
            image_id, coco_format["annotations"], data_info)
        image_id += 1

    with open("./data/{}.json".format("train"), "w") as outfile:
        json.dump(coco_format, outfile, indent=4)
        print("finish!!")

    '''# validation data
    print("preparing val.json")
    image_id = 1
    annotation_id = 1
    coco_format = get_coco_json_format()
    coco_format["categories"] = create_category_annotation(category_ids)
    coco_format["images"] = []
    coco_format["annotations"] = []

    file = allFileList[-1]
    mask_path = f'./nucleus/train/{file}/masks/'
    coco_format["images"], coco_format["annotations"], annotation_id = images_annotations_info(
        mask_path, file, image_id, coco_format["annotations"], coco_format["images"], annotation_id)
    image_id = image_id+1

    with open("./mmdetection/nucleus/{}.json".format("val"), "w") as outfile:
        json.dump(coco_format, outfile, indent=4)
        print("finish!!")'''
