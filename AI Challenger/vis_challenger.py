import cv2
import numpy as np

if __name__ == '__main__':
    """
        "joint_vis": { 1: "visible", 2: "occlude", 3: "invisible"}
        "keypoints":  {
            0: "left shoulder",
            1: "left elbow",
            2: "left wrist",
            3: "right shoulder",
            4: "right elbow",
            5: "right wrist",
            6: "left hip",
            7: "left knee",
            8: "left ankle",
            9: "right hip",
            10: "right knee",
            11: "right ankle",
            12: "head tops"
            13: "upper neck"
        }
        "skeleton": [
            [12,13],[13,0],[0,1],[1,2],[13,3],[3,4],[4,5],
            [13,6],[6,7],[7,8],[13,9],[9,10],[10,11]]
    """

    mode = "check_visible"
    # "full_body" to see keypoints location,
    # or "check_visible" to see vis_annot's meaning

    if mode == "full_body":
        annot = {"url": "http://news.sogou.com/",
                 "image_id": "8776db91659bf1b9abada9bbc9d9f15d0b085642",
                 "keypoint_annotations":
                     {"human1": [400, 183, 1, 380, 319, 1, 358, 416, 1, 559, 217, 1, 557, 357, 1, 590, 294, 1,
                                 428, 444, 1, 394, 664, 1, 0, 0, 3, 511, 445, 1, 504, 662, 1, 0, 0, 3, 466, 51, 1,
                                 485, 179, 1]},
                 "human_annotations": {"human1": [298, 36, 627, 683]}
                 }
    else:
        annot = {"url": "http://news.sogou.com/",
                 "image_id": "043758c591b58f39a01648c49b5154ad1e01d400",
                 "keypoint_annotations": {"human1": [235, 455, 2, 183, 678, 2, 204, 470, 1, 514, 387, 1, 433, 660, 1,
                                                     219, 715, 1, 389, 826, 2, 0, 0, 3, 0, 0, 3, 555, 813, 1, 0, 0, 3,
                                                     0, 0, 3, 191, 183, 1, 324, 395, 1]},
                 "human_annotations": {"human1": [57, 125, 639, 868]}}
    img_name = annot['image_id'] + ".jpg"
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
