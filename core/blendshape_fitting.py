# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 13:35:00 2018

@author: For_Gondor
"""

from core import Blendshape
import numpy as np
from scipy import sparse
from scipy import optimize


def fit_blendshapes_to_landmarks_linear(blendshapes, face_instance, affine_camera_matrix,
                                        landmarks, vertex_ids, lambda_p=500.0):
    """
    Fits blendshape coefficients to given 2D landmarks, given a current face shape instance.
    It's a linear, closed-form solution fitting algorithm, with regularisation (constraining
    the L2-norm of the coefficients). However, there is no constraint on the coefficients,
    so negative coefficients are allowed, which, with linear blendshapes (offsets), will most
    likely not be desired. Thus, prefer the function below.

    This algorithm is very similar to the shape fitting in fit_shape_to_landmarks_linear.
    Instead of the PCA basis, the blendshapes are used, and instead of the mean, a current
    face instance is used to do the fitting from.
    
    Args:
        blendshapes: A vector with blendshapes to estimate the coefficients for.
        face_instance: A shape instance from which the blendshape coefficients should be estimated
        (i.e. the current mesh without expressions, e.g. estimated from a previous PCA-model fitting).
        A 3m x 1 matrix.
        affine_camera_matrix: A 3x4 affine camera matrix from model to screen-space.
        landmarks: 2D landmarks from an image to fit the blendshapes to.
        vertex_ids: The vertex ids in the model that correspond to the 2D points.
        lambda_p: A regularisation parameter, constraining the L2-norm of the coefficients.
    
    Returns:
        The estimated blendshape-coefficients.
    """
    assert len(landmarks) == len(vertex_ids)
    
    num_blendshapes = len(blendshapes)
    num_landmarks = len(landmarks)
    
    # Copy all blendshapes into a "basis" matrix with each blendshape being a column:
    blendshapes_as_basis = Blendshape.to_matrix(blendshapes)
    
    # $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix
    # $V$ associated with the $N$ feature points.
    # And we insert a row of zeros after every third row, resulting in matrix
    # $\hat{V}_h \in R^{4N\times m-1}$:
    v_hat_h = np.zeros([4 * num_landmarks, num_blendshapes])
    for i in range(num_landmarks):
        v_hat_h[i * 4: i * 4 + 3, :] = blendshapes_as_basis[vertex_ids[i] * 3: vertex_ids[i] * 3 + 3, :]
    # Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera
    # matrix C (P_Affine, affine_camera_matrix) is placed on the diagonal:
    p = np.zeros([3 * num_landmarks, 4 * num_landmarks])
    for i in range(num_landmarks):
        p[3 * i: 3 * i + 3, 4 * i: 4 * i + 4] = affine_camera_matrix
    
    # The landmarks in matrix notation (in homogeneous coordinates), $3N\times 1$
    y = np.ones([3 * num_landmarks])
    for i in range(num_landmarks):
        y[3 * i] = landmarks[i][0]
        y[3 * i + 1] = landmarks[i][1]
    # The mean, with an added homogeneous coordinate (x_1, y_1, z_1, 1, x_2, ...)^t
    v_bar = np.ones([4 * num_landmarks])
    for i in range(num_landmarks):
        v_bar[4 * i] = face_instance[vertex_ids[i] * 3]
        v_bar[4 * i + 1] = face_instance[vertex_ids[i] * 3 + 1]
        v_bar[4 * i + 2] = face_instance[vertex_ids[i] * 3 + 2]
    
    # Bring into standard regularised quadratic form:
    a = p.dot(v_hat_h)  # camera matrix times the basis
    b = p.dot(v_bar) - y  # camera matrix times the mean, minus the landmarks
    
    at_a_reg = a.T.dot(a) + lambda_p * np.eye(num_blendshapes)
    rhs = -a.T.dot(b)
    
    coefficients = np.linalg.lstsq(at_a_reg, rhs)[0]
    
    return coefficients


def fit_blendshapes_to_landmarks_nnls(blendshapes, face_instance, affine_camera_matrix,
                                      landmarks, vertex_ids):
    """
    Fits blendshape coefficients to given 2D landmarks, given a current face shape instance.
    Uses non-negative least-squares (NNLS) to solve for the coefficients. The NNLS algorithm
    used doesn't support any regularisation.
    
    This algorithm is very similar to the shape fitting in fit_shape_to_landmarks_linear.
    Instead of the PCA basis, the blendshapes are used, and instead of the mean, a current
    face instance is used to do the fitting from.
    
    Args:
        blendshapes:
            A vector with blendshapes to estimate the coefficients for.
        face_instance:
            A shape instance from which the blendshape coefficients should be
            estimated (i.e. the current mesh without expressions, e.g. estimated from a previous
            PCA-model fitting). A 3m x 1 matrix.
        affine_camera_matrix:
            A 3x4 affine camera matrix from model to screen-space.
        landmarks:
            2D landmarks from an image to fit the blendshapes to.
        vertex_ids:
            The vertex ids in the model that correspond to the 2D points.
        
    Returns:
        The estimated blendshape-coefficients.
    """
    assert len(landmarks) == len(vertex_ids)
    
    num_blendshapes = len(blendshapes)
    num_landmarks = len(landmarks)
    
    # Copy all blendshapes into a "basis" matrix with each blendshape being a column:
    blendshapes_as_basis = Blendshape.to_matrix(blendshapes)
    
    # $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$ associated
    # with the $N$ feature points
    # And we insert a row of zeros after every third row, resulting in matrix $\hat{V}_h \in
    # R^{4N\times m-1}$:
    v_hat_h = np.zeros([4 * num_landmarks, num_blendshapes])
    for i in range(num_landmarks):
        v_hat_h[i * 4: i * 4 + 3, :] = blendshapes_as_basis[vertex_ids[i] * 3: vertex_ids[i] * 3 + 3, :]

    # Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera matrix C
    # (P_Affine, affine_camera_matrix) is placed on the diagonal:
    p_row = []
    p_col = []
    p_data = []
    for i in range(num_landmarks):
        for x in range(np.shape(affine_camera_matrix)[0]):
            for y in range(np.shape(affine_camera_matrix)[1]):
                p_row.append(3 * i + x)
                p_col.append(4 * i + y)
                p_data.append(affine_camera_matrix[x, y])
    p = sparse.coo_matrix((p_data, (p_row, p_col)), shape=(3 * num_landmarks, 4 * num_landmarks))

    # The landmarks in matrix notation (in homogeneous coordinates), $3N\times 1$
    y = np.ones([3 * num_landmarks])
    for i in range(num_landmarks):
        y[3 * i: 3 * i + 2] = landmarks[i][:]
        
    # The mean, with an added homogeneous coordinate (x_1, y_1, z_1, 1, x_2, ...)^t
    v_bar = np.ones([4 * num_landmarks])
    for i in range(num_landmarks):
        v_bar[4 * i: 4 * i + 3] = face_instance[vertex_ids[i] * 3: vertex_ids[i] * 3 + 3]

    # Bring into standard regularised quadratic form:
    a = p.dot(v_hat_h)  # camera matrix times the basis
    b = p.dot(v_bar) - y  # camera matrix times the mean, minus the landmarks

    coefficients = optimize.nnls(a, -b)[0]
    
    return coefficients
