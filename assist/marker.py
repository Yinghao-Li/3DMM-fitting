# -*- coding: utf-8 -*-
"""
Created on Tuesday May 01 12:57:24 2018

@author: For_Gondor
"""

import cv2
import os
import numpy as np
import dlib
from typing import Optional, List


def save_pts(pts: np.ndarray, filename: str) -> None:
    f = open(filename + '.pts', 'w')
    f.write('version: 1\n')
    f.write('n_points: {}\n'.format(len(pts)))
    f.write('{\n')
    for pt in pts:
        f.write('{0:.3f} {1:.3f}\n'.format(pt[0], pt[1]))
    f.write('}\n')
    f.close()
    return None


# 可以读取任意数目的特征点的函数，替换原来的只能读取68个特征点的函数
# 如果需要读取特定数目的特征点，可以在读入之后进行assert操作
def read_pts(filename: str) -> List[List[float]]:
    with open(filename) as f:
        landmarks = []
        a = False
        for line in f:
            if a:
                if '}' in line:
                    break
                coords = line.split()
                landmarks.append([float(coords[0]), float(coords[1])])
            elif '{' in line:
                a = True
    return landmarks


def manual_marker(pic_path: str, num_marks: int = 26) -> None:
    """
    Manually mark a picture. The result will be stored in a pts file with the name of the picture.
    Click mouse left button: Mark a point;
    Click mouse right button: Cancel the last mark.
    ESC: Done. Only functional when the number of marks equals to param 'num_marks'.

    :param pic_path: The full path of picture (including filename).
    :param num_marks: The number of expected marks.
    :return: None.
    """

    line_yellow = np.concatenate(
        [np.ones([11, 1, 1], dtype=np.int8), np.ones([11, 1, 1], dtype=np.int8)*255,
         np.ones([11, 1, 1], dtype=np.int8)*255], axis=2)
    line_blue = np.concatenate(
        [np.ones([1, 11, 1], dtype=np.int8), np.ones([1, 11, 1], dtype=np.int8),
         np.ones([1, 11, 1], dtype=np.int8)*255], axis=2)

    # 定义鼠标事件回调函数
    def on_mouse(event, x, y, flags, params):
        param_marks = params[0]
        param_num_point = params[2]
        param_message_flag = params[1]
        param_pre_point_color = params[3]

        # 鼠标左键按下，设置特征点
        if event == cv2.EVENT_LBUTTONDOWN:
            param_marks.append([x, y])
            param_pre_point_color.append(img[y-5:y+6, x-5:x+6, :].copy())
            img[y-5:y+6, x:x+1, :] = line_yellow.copy()
            img[y:y+1, x-5:x+6, :] = line_blue.copy()
            param_num_point[0] += 1
            param_message_flag[0] = True

        # 鼠标右键，移除特征点
        elif event == cv2.EVENT_RBUTTONDOWN:
            if param_num_point[0] > 0:
                [pre_x, pre_y] = param_marks.pop()
                img[pre_y-5:pre_y+6, pre_x-5:pre_x+6, :] = param_pre_point_color.pop()
                param_num_point[0] -= 1
                param_message_flag[0] = True

        if param_message_flag[0]:
            param_message_flag[0] = False
            print('Point {} has been drawn.'.format(param_num_point[0]))

    # 查找是否存在文件，如果存在则打开
    if os.path.exists(pic_path) and os.path.isfile(pic_path):
        file_name = os.path.basename(pic_path)
        # 对于每一个图片文件重置参数
        num_point = [0]
        marks = []
        pre_point_color = []
        message_flag = [False]

        img = cv2.imread(pic_path)
        width = np.shape(img)[1]
        height = np.shape(img)[0]
        scale_param = 900 / height if height >= width else 900 / width

        img = cv2.resize(img, (round(width * scale_param), round(height * scale_param)), interpolation=cv2.INTER_CUBIC)
        cv2.namedWindow(file_name, cv2.WINDOW_AUTOSIZE)
        # 调用鼠标响应事件
        cv2.setMouseCallback(file_name, on_mouse, [marks, message_flag, num_point, pre_point_color])
        while True:
            cv2.imshow(file_name, img)
            key = cv2.waitKey(20)

            # 如果'esc'被按下且所有特征点标注完成
            if key & 0xFF == 27 and num_point[0] == num_marks:
                break

        cv2.destroyAllWindows()
        marks = np.array(marks, copy=False, dtype=float)
        marks = marks / scale_param

        save_pts(marks, pic_path[:pic_path.rindex('.')])
        return None

    else:
        print('error: file not exist!')
        return None


def frontal_face_marker(pic_path: str, model_path: Optional[str] = r'..\py_share') -> np.ndarray:
    """
    Automatically mark the frontal picture of human face with Dlib Library.
    The result will be stored in a pts file with the name of the picture.

    :param pic_path: The full path of picture (including filename).
    :param model_path: The path of Dlib shape predictor model.
    :return: None
    """

    predictor_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')
    # 使用官方提供的模型构建特征提取器

    predictor = dlib.shape_predictor(predictor_path)

    # 使用dlib自带的frontal_face_detector作为人脸检测器
    detector = dlib.get_frontal_face_detector()

    # 查找是否存在文件，如果存在则打开
    if os.path.exists(pic_path) and os.path.isfile(pic_path):

        img = cv2.imread(pic_path)

        dets = detector(img, 1)

        bbox = dets[0]
        points = predictor(img, bbox)
        marks = np.array([[p.x, p.y] for p in points.parts()], copy=False, dtype=float)

        save_pts(marks, pic_path[:pic_path.rindex('.')])
        return marks

    else:
        print('error: Image file not exist!')


def mark_modifier(pic_path: str, create_duplicate: Optional[bool] = False) -> None:
    """
    Visually changes marks of a picture.

    :param pic_path: The full path of picture (including filename).
    :param create_duplicate: save the changed marks in a new file (original_file_name_new.pts).
    :return: None.
    """

    pts_name = pic_path[:pic_path.rfind('.')] + '.pts'
    # 装载检查过的文件的文件名
    if os.path.isfile(pic_path) and os.path.isfile(pts_name):
        img = cv2.imread(pic_path)
        width = np.shape(img)[1]
        height = np.shape(img)[0]
        scale_param = 900 / height if height >= width else 900 / width
        img = cv2.resize(img, (round(width * scale_param), round(height * scale_param)), interpolation=cv2.INTER_CUBIC)
        pts = (np.array(read_pts(pts_name)) * scale_param).astype(int).tolist()
        num_point = len(pts)
    else:
        if not os.path.isfile(pic_path):
            print('Error: Image file not exist!')
        if not os.path.isfile(pts_name):
            print('Error: PTS file not exist!')
        return None

    line_length = 11
    half_length = 5

    line_yellow = np.concatenate([np.ones([line_length, 1, 1], dtype=np.int8),
                                  np.ones([line_length, 1, 1], dtype=np.int8) * 255,
                                  np.ones([line_length, 1, 1], dtype=np.int8) * 255], axis=2)
    line_red = np.concatenate([np.ones([1, line_length, 1], dtype=np.int8),
                               np.ones([1, line_length, 1], dtype=np.int8),
                               np.ones([1, line_length, 1], dtype=np.int8) * 255], axis=2)

    # 定义鼠标事件回调函数
    def on_mouse(event, x, y, flags, params):
        marks = params[0]
        param_lfirstaction = params[1]
        nth_moving = params[5]
        param_movingpoint = params[2]
        pre_position = params[3]
        param_precolor = params[4]
        param_inicolor = params[6]

        # 鼠标左键按下
        if event == cv2.EVENT_LBUTTONDOWN:
            param_lfirstaction[0] = True
            for i in range(num_point):
                if np.sqrt(np.square(x - marks[0][i][0]) + np.square(y - marks[0][i][1])) < 10:
                    nth_moving[0] = i
                    param_movingpoint[0] = True
                    print("Moving point %d" % i)

        # 鼠标左键抬起
        elif event == cv2.EVENT_LBUTTONUP:
            if param_movingpoint[0]:
                param_inicolor[0][nth_moving[0]] = param_precolor[0].copy()
                print("Point {} was moved to location{}".format(nth_moving[0], (x, y)))
                nth_moving[0] = -1
                param_movingpoint[0] = False

        # 鼠标移动
        elif event == cv2.EVENT_MOUSEMOVE:
            if param_movingpoint[0] and nth_moving[0] != -1:
                if param_lfirstaction[0]:
                    param_precolor[0] = param_inicolor[0][nth_moving[0]].copy()
                    param_lfirstaction[0] = False
                pre_position = marks[0][nth_moving[0]].copy()
                marks[0][nth_moving[0]] = [x, y]
                img[pre_position[1] - half_length:pre_position[1] + half_length + 1,
                    pre_position[0] - half_length:pre_position[0] + half_length + 1, :] = param_precolor[0].copy()

                param_precolor[0] = \
                    img[y - half_length:y + half_length + 1, x - half_length:x + half_length + 1, :].copy()
                img[y - half_length:y + half_length + 1, x:x + 1, :] = line_yellow.copy()
                img[y:y + 1, x - half_length:x + half_length + 1, :] = line_red.copy()

    l_button_first_action = [False]
    moving_point = [False]
    pre_pos = [-1, -1]
    pre_point_color = [np.zeros([line_length, line_length, 3])]
    n_moving = [-1]
    ini_point_color = [np.zeros([num_point, line_length, line_length, 3])]

    for t, p in enumerate(pts):
        ini_point_color[0][t] = \
            img[p[1] - half_length: p[1] + half_length + 1, p[0] - half_length: p[0] + half_length + 1, :].copy()
        img[p[1] - half_length:p[1] + half_length + 1, p[0]:p[0] + 1, :] = line_yellow.copy()
        img[p[1]:p[1] + 1, p[0] - half_length:p[0] + half_length + 1, :] = line_red.copy()
        print("The {}th landmark has coordinates ({},{})".format(t+1, p[0], p[1]))

    file_name = os.path.basename(pic_path)
    cv2.namedWindow(file_name, cv2.WINDOW_AUTOSIZE)

    # 调用鼠标响应事件
    pts = [pts]
    cv2.setMouseCallback(file_name, on_mouse, [pts, l_button_first_action, moving_point, pre_pos, pre_point_color,
                                               n_moving, ini_point_color])
    while True:
        cv2.imshow(file_name, img)
        key = cv2.waitKey(20)

        if key & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    if create_duplicate:
        save_pts(np.array(pts[0], dtype=float)/scale_param, pts_name.split('.')[0] + '_new')
    else:
        save_pts(np.array(pts[0], dtype=float)/scale_param, pts_name.replace('.pts', ''))
    return None
