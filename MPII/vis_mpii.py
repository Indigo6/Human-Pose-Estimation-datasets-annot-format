import numpy as np
import cv2
import os

if __name__ == '__main__':
    """
     (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle,
      6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist,
      11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)
    """
    dst = r"./vised_"
    img_name = r"000003072.jpg"

    roi = [600, 104, 948, 618]
    width = roi[2] - roi[0]
    bigger_times = 480 / width
    big_height = int(bigger_times * (roi[3]-roi[1]))

    label = [[738.0, 538.0], [734.0, 440.0], [717.0, 355.0], [770.0, 355.0],
             [766.0, 443.0], [768.0, 523.0], [744.0, 355.0], [737.0, 216.0], 
             [739.0207, 197.2623], [745.9793, 132.7377], [639.0, 302.0], 
             [684.0, 276.0], [692.0, 217.0], [782.0, 215.0], [805.0, 280.0], 
             [850.0, 308.0]]
    label = [[(label[i][0]-roi[0]) * bigger_times, (label[i][1]-roi[1]) * bigger_times]
             for i in range(len(label))]
    label = np.array(label, dtype=int)
    # print(label)


    img = cv2.imread(img_name)
    img = img[roi[1]:roi[3], roi[0]:roi[2], :]
    img = cv2.resize(img, (480, big_height))

    for i, key in enumerate(label):
        cv2.circle(img, (key[0], key[1]), 6, (0, 0, 255), 2)
        cv2.putText(img, str(i), (key[0], key[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(dst + img_name, img)
    cv2.imshow('src', img)
    cv2.waitKey(0)
