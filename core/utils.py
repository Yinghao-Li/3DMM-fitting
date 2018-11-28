# -*- coding: utf-8 -*-
"""
@author: Yinghao Li
"""

import numpy as np


def clip_to_screen_space(clip_coordinates, screen_width, screen_height):
    """
    Transforms a point from clip space ([-1, 1] x [-1, 1]) to
    image (screen) coordinates, i.e. the window transform.
    Note that the y-coordinate is flipped because the image origin
    is top-left while in clip space top is +1 and bottom is -1.
    No z-division is performed.
    Note: It should rather be called from NDC to screen space?
    
    Exactly conforming to the OpenGL viewport transform, except that
    we flip y at the end.
    Qt: Origin top-left. OpenGL: bottom-left. OCV: top-left.
    
    Args:
        clip_coordinates: A point in clip coordinates.
        screen_width: Width of the screen or window.
        screen_height: Height of the screen or window.
        
    Returns:
        A vector with x and y coordinates transformed to screen space.
    """
    x_ss = (clip_coordinates[0] + 1.0) * (screen_width / 2.0)
    # also flip y; Qt: Origin top-left. OpenGL: bottom-left.
    y_ss = screen_height - (clip_coordinates[1] + 1.0) * (screen_height / 2.0)
    # Note: What we do here is equivalent to x_w = (x *  vW/2) + vW/2;
    # However, Shirley says we should do:x_w = (x *  vW/2) + (vW-1)/2;
    # analogous  for y
    # TODO: Check the consequences.
    return np.array([x_ss, y_ss])


def compute_face_normal(v0, v1, v2):
    """
    Calculates the normal of a face (or triangle), i.e. the
    per-face normal. Return normal will be normalised.
    Assumes the triangle is given in CCW order, i.e. vertices
    in counterclockwise order on the screen are front-facing.
    
    Args:
        v0: First vertex.
        v1: Second vertex.
        v2: Third vertex.
        
    Returns:
        The unit-length normal of the given triangle.
    """
    n = np.cross(v1 - v0, v2 - v0)
    return n / np.linalg.norm(n, axis=1)[:, None]


# 可以读取任意数目的特征点的函数，替换原来的只能读取68个特征点的函数
# 如果需要读取特定数目的特征点，可以在读入之后进行assert操作
def read_pts(filename: str) -> list:
    """
    Read point cloud file that contains face landmarks
    
    Args:
        filename:
            the path of the file
    
    Returns:
        landmarks
    """
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
