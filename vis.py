import argparse
import cv2
import json
import os

import numpy as np

dataset_indexes = {
    'mpii': {
        0: 'right_ankle',
        1: 'right_knee',
        2: 'right_hip',
        3: 'left_hip',
        4: 'left_knee',
        5: 'left_ankle',
        6: 'pelvis',
        7: 'thorax',
        8: 'upper_neck',
        9: 'head_top',
        10: 'right_wrist',
        11: 'right_elbow',
        12: 'right_shoulder',
        13: 'left_shoulder',
        14: 'left_elbow',
        15: 'left_wrist'
    },
    'coco': {
        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'
    },
    'aic': {
        0: "right_shoulder",
        1: "right_elbow",
        2: "right_wrist",
        3: "left_shoulder",
        4: "left_elbow",
        5: "left_wrist",
        6: "right_hip",
        7: "right_knee",
        8: "right_ankle",
        9: "left_hip",
        10: "left_knee",
        11: "left_ankle",
        12: "head_top",
        13: "neck"
    }
}

dataset_skeletons = {
    'mpii': [[10, 11], [11, 12], [15, 14], [14, 13], [0, 1], [1, 2], [5, 4], [4, 3],
             [2, 6], [3, 6], [12, 7], [13, 7], [6, 7], [7, 8], [8, 9]],
    'coco': [[16, 14], [14, 12], [15, 13], [13, 11], [12, 11], [6, 12], [5, 11], [6, 5], [6, 8],
             [8, 10], [5, 7], [7, 9], [3, 1], [1, 0], [0, 2], [2, 4]]
}


def get_roi(my_annot, my_dataset):
    if my_dataset == "mpii":
        center = np.array(my_annot["center"])
        my_scale = np.array([my_annot["scale"] * 200] * 2)
        my_roi = np.append((center - my_scale / 2), my_scale).astype(np.int32)
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
    write_dst_whiteBG = os.path.join(root, r"{}_vised_whiteBG_".format(dataset))
    indexes = dataset_indexes[dataset]
    skeletons = dataset_skeletons[dataset]

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
            cv2.circle(img, (key[0], key[1]), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(img, str(i), (key[0], key[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2,
                        lineType=cv2.LINE_AA)

        cv2.imwrite(write_dst + img_name, img)
        cv2.imshow('src', img)
        cv2.waitKey(0)

        whiteBG_img = np.zeros([h+300, w+300, 3], dtype=np.uint8) + 255
        for i, key in enumerate(label):
            cv2.circle(whiteBG_img, (key[0]+50, key[1]+50), 5, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        for p, q in skeletons:
            cv2.line(whiteBG_img, label[p]+50, label[q]+50, (0, 0, 0), 2, lineType=cv2.LINE_AA)

        cv2.imwrite(write_dst_whiteBG + img_name, whiteBG_img)
        cv2.imshow('src', whiteBG_img)
        cv2.waitKey(0)
