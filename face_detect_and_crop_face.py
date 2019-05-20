"""

We use this program to detect faces in picture by Dlib trained model and functions.
Once faces detected, we crop them and save with extension of jpg.
author: Shuai Zhu @TZX
The following code heavily based on face_detector.py
and the related packages needs to be installed ahead.

"""
import os
import sys
import dlib
import cv2 as cv


## the following detector based on HOG feature.
# sys.argv means the input of you command line
# for example we run .py file as:
#     python face_detector.py
# at this time, sys.argv = [python, face_detector.py]
# in this function we redefine this part by def, such as


def HOG_detector(img_dir, scores=False):
    """
    This function come from Dlib python example. python_exampls/face_detector.py
    :param img_dir: imput image directionay
    :param scores:  whether output scores
    :return: img: full image, dets: Dlib class store the rectangle info. of crop
    """
    detector = dlib.get_frontal_face_detector()
    # win = dlib.image_window()
    if scores == False:
        img = dlib.load_rgb_image(img_dir)
        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces.
        dets = detector(img, 1)
        #### ======================================
        ####    if image need to be shown
        #### ======================================
        # print("Number of faces detected: {}".format(len(dets)))
        # for i, d in enumerate(dets):
        #     print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #         i, d.left(), d.top(), d.right(), d.bottom()))
        # win.clear_overlay()
        # win.set_image(img)
        # win.add_overlay(dets)
        # dlib.hit_enter_to_continue()
    else:
        img = dlib.load_rgb_image(img_dir)
        dets, scores, idx = detector.run(img, 1, -1)
        # for i, d in enumerate(dets):
        #     print("Detection {}, score: {}, face_type:{}".format(
        #         d, scores[i], idx[i]))
    return img, dets


def crop_save_faces(img_dir,dets):
    """
    :param img: The input full image
    :param dets:  The cropped data
    :return: folder named after the img name and contains the cropped faces inside
    """
    img = dlib.load_rgb_image(img_dir)
    up_folder = os.path.dirname(img_dir)
    img_name = os.path.split(img_dir)[-1].split('.')[0]
    name_folder = up_folder + '/' + img_name
    if not os.path.exists(name_folder):
        os.mkdir(name_folder)

    for i, sub_img in enumerate(dets):
        crop = img[sub_img.top():sub_img.bottom(), sub_img.left():sub_img.right()]
        sub_img_save = os.path.join(name_folder, img_name) + '_' + str(i) + '.jpg'
        cv.imwrite(sub_img_save, cv.cvtColor(crop, cv.COLOR_RGB2BGR))


if __name__ == '__main__':
    img, dets = HOG_detector(sys.argv[1])
    crop_save_faces(sys.argv[1], dets)
