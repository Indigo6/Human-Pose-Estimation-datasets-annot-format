import numpy as np
import cv2
import os

if __name__ == '__main__':
    """
        "joint_vis": { 0: "invisible", 1: "occlude", 2: "visible" }
        "keypoints": {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
            4: "right_ear",
            5: "left_shoulder",
            6: "right_shoulder",
            7: "left_elbow",
            8: "right_elbow",
            9: "left_wrist",
            10: "right_wrist",
            11: "left_hip",
            12: "right_hip",
            13: "left_knee",
            14: "right_knee",
            15: "left_ankle",
            16: "right_ankle"
        },
        "skeleton": [
            [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
            [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    """
    dst = r"./vised_"

    mode = "check_visible"
    # "full_body" to see keypoints location,
    # or "check_visible" to see vis_annot's meaning

    if mode == "full_body":
        img_name = r"000000017905.jpg"
        label = [132, 261, 2, 139, 254, 2, 126, 254, 2, 147, 257, 2, 117, 258, 2, 171, 299, 2,
                 104, 303, 2, 177, 351, 2, 98, 360, 2, 182, 398, 2, 94, 414, 2, 164, 406, 2,
                 121, 407, 2, 170, 489, 2, 124, 488, 2, 182, 570, 2, 124, 569, 2]
        roi = [56, 220, 220, 608]
    else:
        img_name = r"000000285138.jpg"
        label = [268, 204, 2, 308, 177, 2, 250, 171, 2, 341, 186, 2, 220, 174, 2, 401, 337, 2,
                 162, 276, 1, 399, 502, 2, 48, 417, 2, 273, 584, 2, 56, 292, 2, 284, 629, 2,
                 148, 599, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        roi = [20, 60, 435, 640]

    width = roi[2] - roi[0]
    bigger_times = 480 / width
    big_height = int(bigger_times * (roi[3] - roi[1]))

    label = [[(label[i] - roi[0]) * bigger_times, (label[i + 1] - roi[1]) * bigger_times]
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
