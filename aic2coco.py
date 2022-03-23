import json
import copy

aic_train_path = "aic/annotations/aic_train.json"
aic_val_path = "aic/annotations/aic_val.json"
coco_train_path = "coco/annotations/person_keypoints_train2017.json"
coco_val_path = "coco/annotations/person_keypoints_val2017.json"

with open(aic_train_path, 'r') as f:
    aic_train = json.load(f)

with open(aic_val_path, 'r') as f:
    aic_val = json.load(f)

with open(coco_train_path, 'r') as f:
    coco_train = json.load(f)

with open(coco_val_path, 'r') as f:
    coco_val = json.load(f)

extra2coco = [6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, -1, -1]
coco2extra = [-1, -1, -1, -1, -1, 3, 0, 4, 1, 5, 2, 9, 6, 10, 7, 11, 8]


aic_train['categories'] = coco_train['categories']
for i in range(len(aic_train['images'])):
    aic_train['images'][i]['id'] += 600000
for i in range(len(aic_train['annotations'])):
    aic_keypoints = copy.deepcopy(aic_train['annotations'][i]['keypoints'])
    aic_train['annotations'][i]['keypoints'] = [0] * 51
    for j in range(5, 17):
        aic_idx = coco2extra[j]
        aic_train['annotations'][i]['keypoints'][3*j] = aic_keypoints[3*aic_idx]
        aic_train['annotations'][i]['keypoints'][3*j+1] = aic_keypoints[3*aic_idx+1]
        aic_train['annotations'][i]['keypoints'][3*j+2] = aic_keypoints[3*aic_idx+2]
    aic_train['annotations'][i]['image_id'] += 600000
    aic_train['annotations'][i]['id'] += 900100582000

with open('aic/annotations/aic_train_coco_format.json', 'w+') as f:
    json.dump(aic_train, f)

for i in range(len(aic_val['images'])):
    aic_val['images'][i]['id'] += 600000 + 210000
aic_val['categories'] = coco_train['categories']
for i in range(len(aic_val['annotations'])):
    aic_keypoints = copy.deepcopy(aic_val['annotations'][i]['keypoints'])
    aic_val['annotations'][i]['keypoints'] = [0] * 51
    for j in range(5, 17):
        aic_idx = coco2extra[j]
        aic_val['annotations'][i]['keypoints'][3*j] = aic_keypoints[3*aic_idx]
        aic_val['annotations'][i]['keypoints'][3*j+1] = aic_keypoints[3*aic_idx+1]
        aic_val['annotations'][i]['keypoints'][3*j+2] = aic_keypoints[3*aic_idx+2]
    aic_val['annotations'][i]['image_id'] += 600000 + 210000
    aic_val['annotations'][i]['id'] += 900100582000 + 210000

with open('aic/annotations/aic_val_coco_format.json', 'w+') as f:
    json.dump(aic_val, f)

coco_train['images'].extend(aic_train['images'])
coco_train['annotations'].extend(aic_train['annotations'])
print("+ aic train: {} samples.".format(len(coco_train['annotations'])))
with open('coco/annotations/coco_train_aic_train.json', 'w+') as f:
    json.dump(coco_train, f)

coco_train['images'].extend(aic_val['images'])
coco_train['annotations'].extend(aic_val['annotations'])
print("+ aic train and val: {} samples.".format(len(coco_train['annotations'])))
with open('coco/annotations/coco_train_aic_trainval.json', 'w+') as f:
    json.dump(coco_train, f)

print('test')