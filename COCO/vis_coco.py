import numpy as np
import cv2
import os

if __name__ == '__main__':
    dst = r"./vised_"
    roi = [56, 220, 220, 608]
    width = roi[2] - roi[0]
    bigger_times = 480 / width
    big_height = int(bigger_times * (roi[3]-roi[1]))

    img_name = r"000000017905.jpg"
    label = [132,261,2,139,254,2,126,254,2,147,257,2,117,258,2,171,299,2,104,303,2,177,
             351,2,98,360,2,182,398,2,94,414,2,164,406,2,121,407,2,170,489,2,124,488,2,
             182,570,2,124,569,2]
    label = [[(label[i]-roi[0]) * bigger_times, (label[i+1]-roi[1])* bigger_times]
             for i in range(0, len(label), 3)]
    label = np.array(label, dtype=int)
    # print(label)

    img = cv2.imread(img_name)
    img = img[roi[1]:roi[3], roi[0]:roi[2], :]
    img = cv2.resize(img, (480, big_height))

    for i, key in enumerate(label):
        cv2.circle(img, (key[0], key[1]), 3, (0, 0, 255), 2)
        cv2.putText(img, str(i), (key[0], key[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imwrite(dst + img_name, img)
    cv2.imshow('src', img)
    cv2.waitKey(0)
