# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 11:24:01 2018

@author: For_Gondor
"""

import json
import numpy as np
from core import glm
from enum import IntEnum


class Frustum:
    """
    A class representing a camera viewing frustum. At the moment only fully tested with orthographic camera.
    
    Attributes:
        l: left
        r: right
        b: bottom
        t: top
    """
    def __init__(self, l=-1.0, r=1.0, b=-1.0, t=1.0):
        """
        initialize Frustum class
        """
        self.l = l
        self.r = r
        self.b = b
        self.t = t


class CameraType(IntEnum):
    """
    Just realize # define function in C++
    """
    Orthographic = 1
    Perspective = 2


class RenderingParameters:
    """
    Represents a set of estimated model parameters (rotation, translation) and
    camera parameters (viewing frustum).
    
    The estimated rotation and translation transform the model from model-space to camera-space,
    and, if one wishes to use OpenGL, can be used to build the model-view matrix.
    The parameters are the inverse of the camera position in 3D space.
    
    The camera frustum describes the size of the viewing plane of the camera, and
    can be used to build an OpenGL-conformant orthographic projection matrix.
    
    Together, these parameters fully describe the imaging process of a given model instance
    (under an orthographic projection).
    
    The rotation values are given in radians and estimated using the RPY convention.
    Yaw is applied first to the model, then pitch, then roll (R * P * Y * vertex).
    In general, the convention is as follows:
        r_x = Pitch
        r_y = Yaw. Positive means subject is looking left (we see her right cheek).
        r_z = Roll. Positive means the subject's right eye is further down than the other one (he
        tilts his head to the right). However, we're using a quaternion now to represent the rotation, and
        glm::eulerAngles() will give slightly different angles (according to a different (undocumented))
        convention. However, the rotation is exactly the same! (i.e. they are represented by the same
        quaternion / rotation matrix).
        
    This should always represent all parameters necessary to render the model to an image, and be
    completely OpenGL compliant.
    """
    def __init__(self, ortho_params, screen_width, screen_height):
        """
        Initialize RenderingParameters class
        
        This assumes estimate_sop was run on points with OpenCV viewport! I.e. y flipped.
        """
        self.__camera_type = CameraType['Orthographic']
        self.__t_x = ortho_params.tx
        self.__t_y = ortho_params.ty
        self.__screen_width = screen_width
        self.__screen_height = screen_height
        self.__rotation = glm.quat_cast(ortho_params.R)
        self.__frustum = Frustum(0.0, screen_width / ortho_params.s, 0.0, screen_height / ortho_params.s)
    
    def get_camera_type(self):
        return self.__camera_type
    
    def get_rotation(self):
        return self.__rotation
    
    def set_rotation(self, rotation_quaternion):
        self.__rotation = rotation_quaternion
        return None
    
    def get_translation(self):
        return self.__t_x, self.__t_y
    
    def set_translation(self, t_x, t_y):
        self.__t_x = t_x
        self.__t_y = t_y
        return None
    
    def get_modelview(self):
        modelview = glm.mat4_cast(self.__rotation)
        modelview[0, 3] = self.__t_x
        modelview[1, 3] = self.__t_y
        return modelview
    
    def get_projection(self):
        if self.__camera_type == CameraType['Orthographic']:
            return glm.ortho(self.__frustum.l, self.__frustum.r, self.__frustum.b, self.__frustum.t)
        else:
            raise RuntimeError("get_projection() for CameraType::Perspective is not implemented yet.")
    
    def get_frustum(self):
        return self.__frustum
    
    def set_frustum(self, frustum):
        self.__frustum = frustum
        return None
    
    def get_screen_width(self):
        return self.__screen_width
    
    def set_screen_width(self, screen_width):
        self.__screen_width = screen_width
        return None
    
    def get_screen_height(self):
        return self.__screen_height
    
    def set_screen_height(self, screen_height):
        self.__screen_height = screen_height
        return None


def save_rendering_parameters(rendering_parameters, filename):
    """
    Saves the rendering parameters for an image to a json file.
    
    Args:
        rendering_parameters: An instance of class RenderingParameters.
        filename: The file to write.
        
    Returns:
        None
    """
    outer_dict = dict()
    inner_dict = dict()
    inner_dict['camera_type'] = rendering_parameters.get_camera_type()
    inner_dict['frustum'] = rendering_parameters.get_frustum().__dict__
    inner_dict['rotation'] = rendering_parameters.get_rotation().__dict__
    inner_dict['t_x'] = rendering_parameters.get_translation()[0]
    inner_dict['t_y'] = rendering_parameters.get_translation()[1]
    inner_dict['screen_width'] = rendering_parameters.get_screen_width()
    inner_dict['screen_height'] = rendering_parameters.get_screen_height()
    outer_dict['rendering_parameters'] = inner_dict
    
    file = open(filename, 'w')
    json.dump(outer_dict, file, indent=4)
    file.close()
    
    return None


def get_opencv_viewport(width, height):
    """
    Returns a glm/OpenGL compatible viewport vector that flips y and
    has the origin on the top-left, like in OpenCV.
    """
    return np.array([0, height, width, -height])


def get_3x4_affine_camera_matrix(params, width, height):
    """
    Creates a 3x4 affine camera matrix from given fitting parameters. The
    matrix transforms points directly from model-space to screen-space.
    
    This function is mainly used since the linear shape fitting fitting::fit_shape_to_landmarks_linear
    expects one of these 3x4 affine camera matrices, as well as render::extract_texture.
    
    Args:
        params: RenderingParameters object
        width: int object
        height: int object
        
    Returns:
        a 3x4 ndarray
    """
    view_model = params.get_modelview()
    ortho_projection = params.get_projection()
    mvp = ortho_projection.dot(view_model)
    
    # TODO: redundant defination
    viewport = [0, height, width, -height]  # flips y, origin top-left, like in OpenCV
    # equivalent to what glm::project's viewport does, but we don't change z and w:
    viewport_mat = np.eye(4)
    viewport_mat[0, 0] = viewport[2] / 2.0
    viewport_mat[0, 3] = viewport[2] / 2.0 + viewport[0]
    viewport_mat[1, 1] = viewport[3] / 2.0
    viewport_mat[1, 3] = viewport[3] / 2.0 + viewport[1]
    
    full_projection_4x4 = viewport_mat.dot(mvp)
    # we take the first 3 rows, but then set the last one to [0 0 0 1]
    full_projection_3x4 = full_projection_4x4[:3, :]
    full_projection_3x4[2, :] = np.array([0.0, 0.0, 0.0, 1.0])
    
    return full_projection_3x4
