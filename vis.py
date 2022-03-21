import argparse
import cv2
import json
import os

import numpy as np


def get_roi(my_annot, my_dataset):
    if my_dataset == "mpii":
        center = np.array(my_annot["center"])
        my_scale = np.array([my_annot["scale"] * 200] * 2)
        my_roi = np.append((center - my_scale/2), my_scale).astype(np.int32)
    else:
        my_roi = np.array(my_annot["bbox"], dtype=np.int32)
    return my_roi


def get_suitable_wh(my_roi):
    screen = np.array([1600, 900])
    my_scale = min(screen / roi[2:])
    return int(my_roi[2] * my_scale), int(my_roi[3] * my_scale), my_scale


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Dataset Annotation Format')
    # general
    parser.add_argument('--name',
                        help='dataset name',
                        default="mpii",
                        choices=["mpii", "coco", "aic", "aic_coco"])

    args = parser.parse_args()
    dataset = args.name

    dataset_roots = {
        "mpii": "MPII/",
        "coco": "COCO/",
        "aic": "AI Challenger/",
        "aic_coco": "AI Challenger/"
    }

    dataset_annots = {
        "mpii": "mpii.json",
        "coco": "coco.json",
        "aic": "aic.json",
        "aic_coco": "aic_coco_format.json"
    }

    root = dataset_roots[dataset]
    annot_file = os.path.join(root, dataset_annots[dataset])
    write_dst = os.path.join(root, r"{}_vised_".format(dataset))

    with open(annot_file) as f:
        annots = json.load(f)

    if dataset != "mpii":
        id2name = {}
        for image_info in annots['images']:
            id2name[image_info['id']] = image_info['file_name']
        annotations = annots['annotations']
    else:
        annotations = annots

    for annot in annotations:
        roi = get_roi(annot, dataset)
        w, h, scale = get_suitable_wh(roi)

        if dataset == "mpii":
            img_name = annot["image"]
            label = annot['joints']
            label = [[(item[0] - roi[0]) * scale, (item[1] - roi[1]) * scale]
                     for item in label]
        else:
            img_name = id2name[annot['image_id']]
            label = annot['keypoints']
            label = [[(label[i] - roi[0]) * scale, (label[i + 1] - roi[1]) * scale]
                     for i in range(0, len(label), 3)]

        label = np.array(label, dtype=np.int32)
        # print(label)

        img = cv2.imread(os.path.join(root, img_name))
        img = img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]
        img = cv2.resize(img, (w, h))

        for i, key in enumerate(label):
            cv2.circle(img, (key[0], key[1]), 3, (0, 0, 255), 2)
            cv2.putText(img, str(i), (key[0], key[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        cv2.imwrite(write_dst + img_name, img)
        cv2.imshow('src', img)
        cv2.waitKey(0)
