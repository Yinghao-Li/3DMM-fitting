# -*- coding: utf-8 -*-
"""
@author: Yinghao Li
"""

import numpy as np


class ScaledOrthoProjectionParameters:
    """
    Parameters of an estimated scaled orthographic projection.
    
    Attributes:
        R: Euler rotation matrix, a 3x3 matrix, represented by ndarray
        tx, ty: Translation x and translation y
        s: Scaling
    """
    def __init__(self, r=np.empty([3, 3]), tx=0.0, ty=0.0, s=0.0):
        self.R = r
        self.tx = tx
        self.ty = ty
        self.s = s


def estimate_orthographic_projection_linear(image_points, model_points, is_viewport_upsidedown, viewport_height=None):
    """
    Estimates the parameters of a scaled orthographic projection.
    
    Given a set of 2D-3D correspondences, this algorithm estimates rotation,
    translation (in x and y) and a scaling parameters of the scaled orthographic
    projection model using a closed-form solution. It does so by first computing
    an affine camera matrix using algorithm [1], and then finds the closest
    orthonormal matrix to the estimated affine transform using SVD.
    This algorithm follows the original implementation [2] of William Smith,
    University of York.
    
    Requires >= 4 corresponding points.
    
    [1]: Gold Standard Algorithm for estimating an affine camera matrix from
    world to image correspondences, Algorithm 7.2 in Multiple View Geometry,
    Hartley & Zisserman, 2nd Edition, 2003.
    [2]: https://github.com/waps101/3DMM_edges/blob/master/utils/POS.m
    
    Args:
        image_points:
            A list of 2D image points, with the shape of nx2.
        model_points:
            Corresponding points of a 3D model, with the shape of nx3.
        is_viewport_upsidedown:
            Flag to set whether the viewport of the image points is upside-down (e.g. as in OpenCV).
        viewport_height:
            Height of the viewport of the image points (needs to be given if is_viewport_upsidedown == true).
        
    Returns:
        Rotation, translation and scaling of the estimated scaled orthographic projection.
    """
    assert len(image_points) == len(model_points)
    # Number of correspondence points given needs to be equal to or larger than 4
    assert len(image_points) >= 4
    
    num_correspondences = len(image_points)
    
    image_points = np.array(image_points)
    model_points = np.array(model_points)
    
    # TODO: Might be problematic, should be noticed!
    if is_viewport_upsidedown:
        if viewport_height is None:
            raise RuntimeError('Error: If is_viewport_upsidedown is set to true, viewport_height needs to be given.')
        for ip in image_points:
            ip[1] = viewport_height - ip[1]
    
    # Build linear system of equations in 8 unknowns of projection matrix
    a = np.zeros([2 * num_correspondences, 8])
    # !! This part was wrong, and has been corrected.
    a[0: 2 * num_correspondences: 2, :4] = model_points.copy()
    a[1: 2 * num_correspondences: 2, 4:] = model_points.copy()
    
    # TODO: Is it necessary?
    b = np.reshape(image_points, [2 * num_correspondences])
    
    # Using pseudo-inverse matrix (sdv) to solve linear system
    k = np.linalg.lstsq(a, b)[0]
    
    # Extract params from recovered vector
    r_1 = k[0:3]
    r_2 = k[4:7]
    stx = k[3]
    sty = k[7]
    s = (np.linalg.norm(r_1) + np.linalg.norm(r_2)) / 2
    r1 = r_1 / np.linalg.norm(r_1)
    r2 = r_2 / np.linalg.norm(r_2)
    r3 = np.cross(r1, r2)
    r = np.array([r1, r2, r3])
    
    # Set R_ortho to closest orthogonal matrix to estimated rotation matrix
    [u, _, vt] = np.linalg.svd(r)
    r_ortho = u.dot(vt)
    
    # The determinant of r must be 1 for it to be a valid rotation matrix
    if np.linalg.det(r_ortho) < 0:
        u[2, :] = -u[2, :]
        r_ortho = u.dot(vt)
    
    # Remove the scale from the translations
    t1 = stx / s
    t2 = sty / s
    
    return ScaledOrthoProjectionParameters(r_ortho, t1, t2, s)
