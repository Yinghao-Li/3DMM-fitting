# -*- coding: utf-8 -*-
"""
Created on Sun May 6 13:55:03 2018

@author: For_Gondor
"""

import os
import numpy as np
import cv2
import sys

sys.path.append('..')
from core import Blendshape, contour_correspondence, EdgeTopology, fitting, LandmarkMapper, Landmark, MorphableModel, \
    utils, RenderingParameters, render
# from assist import marker

frontal_pic_name = '00029ba010_960521'
profile_pic_name = '00029pr010_940128'
frontal_img = cv2.imread(os.path.join(r'..\data', frontal_pic_name + '.tif'))
profile_img = cv2.imread(os.path.join(r'..\data', profile_pic_name + '.tif'))
width = np.shape(frontal_img)[1]
height = np.shape(frontal_img)[0]
# marker.frontal_face_marker(os.path.join(r'..\data', frontal_pic_name + '.tif'))

# s = 2
s = 2000 / height if height >= width else 2000 / width
scale_param = 900 / height if height >= width else 900 / width
# scale_param = 1

py_model = MorphableModel.load_model(r"..\py_share\py_sfm_shape_3448.bin")
py_blendshapes = Blendshape.load_blendshapes(r"..\py_share\py_expression_blendshapes_3448.bin")
py_landmark_mapper = LandmarkMapper.LandmarkMapper(r'..\py_share\ibug_to_sfm.txt')
py_edge_topology = EdgeTopology.load_edge_topology(r'..\py_share\py_sfm_3448_edge_topology.json')
py_contour_landmarks = contour_correspondence.ContourLandmarks()
py_contour_landmarks.load(r'..\py_share\ibug_to_sfm.txt')
py_model_contour = contour_correspondence.ModelContour()
py_model_contour.load(r'..\py_share\sfm_model_contours.json')
profile_landmark_mapper = LandmarkMapper.ProfileLandmarkMapper(r'..\py_share\profile_to_sfm.txt')

frontal_landmarks = []
landmark_ids = list(map(str, range(1, 69)))  # generates the numbers 1 to 68, as strings
landmarks = utils.read_pts(os.path.join(r'..\data', frontal_pic_name + '.pts'))
for i in range(68):
    frontal_landmarks.append(Landmark.Landmark(landmark_ids[i], [landmarks[i][0] * s, landmarks[i][1] * s]))

profile_landmarks = []
landmarks = utils.read_pts(os.path.join(r'..\data', profile_pic_name + '.pts'))
for x in profile_landmark_mapper.right_mapper.keys():
    coor = landmarks[int(x) - 1]
    profile_landmarks.append(Landmark.Landmark(x, [coor[0] * s, coor[1] * s]))

py_mesh, frontal_rendering_params, profile_rendering_params = fitting.fit_front_and_profile(
    py_model, py_blendshapes, frontal_landmarks, py_landmark_mapper, profile_landmarks, profile_landmark_mapper,
    round(width * s), round(height * s), py_edge_topology, py_contour_landmarks, py_model_contour, lambda_p=20,
    num_iterations=10)

profile_img = cv2.resize(profile_img, (round(width * scale_param), round(height * scale_param)),
                         interpolation=cv2.INTER_CUBIC)
render.draw_wireframe_with_depth(
    profile_img, py_mesh, profile_rendering_params.get_modelview(), profile_rendering_params.get_projection(),
    RenderingParameters.get_opencv_viewport(width * s, height * s), profile_landmark_mapper, scale_param / s)

frontal_img = cv2.resize(frontal_img, (round(width * scale_param), round(height * scale_param)),
                         interpolation=cv2.INTER_CUBIC)
render.draw_wireframe_with_depth(
    frontal_img, py_mesh, frontal_rendering_params.get_modelview(), frontal_rendering_params.get_projection(),
    RenderingParameters.get_opencv_viewport(width * s, height * s), py_landmark_mapper, scale_param / s)

for lm in frontal_landmarks:
    cv2.rectangle(
        frontal_img, (int(lm.coordinates[0] * scale_param / s) - 2, int(lm.coordinates[1] * scale_param / s) - 2),
        (int(lm.coordinates[0] * scale_param / s) + 2, int(lm.coordinates[1] * scale_param / s) + 2), (255, 0, 0))

for lm in profile_landmarks:
    cv2.rectangle(
        profile_img, (int(lm.coordinates[0] * scale_param / s) - 2, int(lm.coordinates[1] * scale_param / s) - 2),
        (int(lm.coordinates[0] * scale_param / s) + 2, int(lm.coordinates[1] * scale_param / s) + 2), (255, 0, 0))

img = np.hstack([frontal_img, profile_img])
cv2.imwrite(frontal_pic_name + '-outcome.jpg', img)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

render.save_ply(py_mesh, frontal_pic_name + '-output', [210, 183, 108], author='Yinghao Li')
