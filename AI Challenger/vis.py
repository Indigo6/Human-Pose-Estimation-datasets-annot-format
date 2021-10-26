import cv2
import numpy as np

if __name__ == '__main__':
    # img_name = "c681fcce2fab08692d11100fd8195353cf27a631.jpg"
    # annot = {"url": "http://news.sogou.com/", "image_id": "c681fcce2fab08692d11100fd8195353cf27a631", "keypoint_annotations": {"human1": [469, 181, 1, 499, 290, 1, 493, 342, 1, 373, 178, 1, 356, 280, 1, 361, 309, 1, 462, 378, 1, 468, 506, 1, 481, 622, 1, 400, 382, 1, 397, 507, 1, 394, 621, 1, 368, 115, 1, 402, 156, 1]}, "human_annotations": {"human1": [333, 98, 511, 667]}}
    annot = {"url": "http://news.sogou.com/", "image_id": "8776db91659bf1b9abada9bbc9d9f15d0b085642", "keypoint_annotations": {"human1": [400, 183, 1, 380, 319, 1, 358, 416, 1, 559, 217, 1, 557, 357, 1, 590, 294, 1, 428, 444, 1, 394, 664, 1, 0, 0, 3, 511, 445, 1, 504, 662, 1, 0, 0, 3, 466, 51, 1, 485, 179, 1]}, "human_annotations": {"human1": [298, 36, 627, 683]}}
    img_name = "8776db91659bf1b9abada9bbc9d9f15d0b085642.jpg"
    dst = r"./vised_"

    img = cv2.imread(img_name)

    roi = annot['human_annotations']['human1']
    width = roi[2] - roi[0]
    bigger_times = 480 / width
    big_height = int(bigger_times * (roi[3]-roi[1]))

    label = []
    for i in range(0,len(annot['keypoint_annotations']['human1']),3):
        label.append([annot['keypoint_annotations']['human1'][i],annot['keypoint_annotations']['human1'][i+1]])
    label = [[(label[i][0]-roi[0]) * bigger_times, (label[i][1]-roi[1]) * bigger_times]
             for i in range(len(label))]
    label = np.array(label, dtype=int)
    # print(label)


    img = cv2.imread(img_name)
    img = img[roi[1]:roi[3], roi[0]:roi[2], :]
    img = cv2.resize(img, (480, big_height))

    for i, key in enumerate(label):
        if np.all(key == 0):
            continue
        cv2.circle(img, (key[0], key[1]), 6, (0, 0, 255), 2)
        cv2.putText(img, str(i), (key[0], key[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(dst + img_name, img)
    cv2.imshow('src', img)
    cv2.waitKey(0)
