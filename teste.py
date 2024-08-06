import json
import os
import cv2
import numpy as np

# folder path
dir_path = r'C:/Users/renan/Desktop/MMP/dataset/annotations/render'

# list to store files
files = []

# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        files.append(path)

key = ['head_top',
'left_heel',        
'right_heel',           
'crotch',
'left_shoulder',
'right_shoulder',
'left_hand',
'right_hand',
'head_leftmost',
'head_rightmost',
'neck_leftmost',
'neck_rightmost',
'chest_leftmost',
'chest_rightmost',
'waist_leftmost',
'waist_rightmost',
'hip_leftmost',
'hip_rightmost',
'wrist_leftmost',
'wrist_rightmost',
'bicep_leftmost',
'bicep_rightmost',
'forearm_leftmost',
'forearm_rightmost',
'thigh_leftmost',
'thigh_rightmost',
'calf_leftmost',
'calf_rightmost',
'ankle_leftmost',
'ankle_rightmost']

key_front = ['head_top',
'left_heel',        
'right_heel',           
'crotch',
'left_shoulder',
'right_shoulder',
'left_hand',
'right_hand',
'head_leftmost',
'head_rightmost',
'neck_leftmost',
'neck_rightmost',
'chest_leftmost',
'chest_rightmost',
'waist_leftmost',
'waist_rightmost',
'hip_leftmost',
'hip_rightmost',
'wrist_leftmost',
'wrist_rightmost',
'bicep_leftmost',
'bicep_rightmost',
'forearm_leftmost',
'forearm_rightmost',
'thigh_leftmost',
'thigh_rightmost',
'calf_leftmost',
'calf_rightmost',
'ankle_leftmost',
'ankle_rightmost']

key_side = ['head_top',
'left_heel',        
'right_heel',
'head_leftmost',
'head_rightmost',
'neck_leftmost',
'neck_rightmost',
'chest_leftmost',
'chest_rightmost',
'waist_leftmost',
'waist_rightmost',
'hip_leftmost',
'hip_rightmost',
'wrist_leftmost',
'wrist_rightmost',
'bicep_leftmost',
'bicep_rightmost',
'forearm_leftmost',
'forearm_rightmost',
'thigh_leftmost',
'thigh_rightmost',
'calf_leftmost',
'calf_rightmost',
'ankle_leftmost',
'ankle_rightmost']

info = {"info":
    {"description": "Body Measurement Dataset",
    "version": "1.0",
    "year": 2024},

    "licenses":[
    {
        "url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "id": 1,
        "name": "CC0 1.0 UNIVERSAL"
    }],

    "images":[],
    "annotations":[],
    "categories": [{"supercategory": "person","id": 1,"name": "person","keypoints": [key],"skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]}]
    }


with open('annotation_render.json','w') as ann:
    json.dump(info,ann)


ann = open("annotation_render.json")
json_ann = json.load(ann)
count =0
for nome in files:
    keypoints = []
    
    f = open("C:/Users/renan/Desktop/MMP/dataset/annotations/render/"+nome)
    data = json.load(f)

    for x in data['projections']['landmarks']:
        if x in key:
            keypoints.append(data['projections']['landmarks'][x][0])
            keypoints.append(data['projections']['landmarks'][x][1])
            if data["view"] == "side" and x in key_side:
                keypoints.append(2)
            elif data["view"] == "front" and x in key_front:
                keypoints.append(2)
            else:
                keypoints.append(1)
    for x in data['projections']['joints']:
        if x in key:
            keypoints.append(data['projections']['joints'][x][0])
            keypoints.append(data['projections']['joints'][x][1])
            if data["view"] == "side" and x in key_side:
                keypoints.append(2)
            elif data["view"] == "front" and x in key_front:
                keypoints.append(2)
            else:
                keypoints.append(1)


    for x in data['projections']['extremes']:
        for y in data['projections']['extremes'][x]:
            if x+"_"+y in key:
                keypoints.append(data['projections']['extremes'][x][y][0])
                keypoints.append(data['projections']['extremes'][x][y][1])
                if data["view"] == "side" and x in key_side:
                    keypoints.append(2)
                elif data["view"] == "front" and x in key_front:
                    keypoints.append(2)
                else:
                    keypoints.append(1)
            
    if data["view"] == "frontal":
        image = cv2.imread("C:/Users/renan/Desktop/MMP/dataset/train/frontal/train_"+data["file_numeration"]+"_frontal_N_1.png")
    if data["view"] == "side":
        image = cv2.imread("C:/Users/renan/Desktop/MMP/dataset/train/side/train_"+data["file_numeration"]+"_side_N_1.png")

    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    out = np.zeros_like(gray)
    x,y,w,h = cv2.boundingRect(contours[0])
    point1 = (x,y)
    point2 = (x+w,y)
    point3 = (x,y+h)
    point4 = (x+w,y+h)

    im_part = {
        "license": 1,
        "file_name": "train_render_"+str(data["file_numeration"])+"_"+str(data["view"])+"_N_1_render.png",
        "height": 1024,
        "width": 1024,
        "date_captured": "2024-12-12 00:00:00",
        "id": data["file_numeration"]
    }

    ann_part = {"segmentation":[contours],
        "num_keypoints": 32,
        "area": 1048576,
        "iscrowd": 0,
        "keypoints": keypoints,
        "image_id": data["file_numeration"],
        "bbox": [point1,point2,point3,point4],
        "category_id": 1,
        "id": data["file_numeration"]}
    
    json_ann["images"].append(im_part)
    json_ann["annotations"].append(ann_part)
    count+=1
    if count>=4:
        break


with open('annotation_render.json','w') as ann:
    json.dump(json_ann,ann)
