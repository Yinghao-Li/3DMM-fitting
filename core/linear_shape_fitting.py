# -*- coding: utf-8 -*-
"""
@author: Yinghao Li
"""

import numpy as np
from scipy import sparse


def fit_shape_to_landmarks_linear(shape_model, affine_camera_matrix, landmarks, vertex_ids,
                                  base_face=list(), lambda_p=3.0, num_coefficients_to_fit=None,
                                  detector_standard_deviation=3.0 ** 0.5, model_standard_deviation=0.0):
    """
    Fits the shape of a Morphable Model to given 2D landmarks (i.e. estimates the maximum likelihood
    solution of the shape coefficients) as proposed in [1].
    It's a linear, closed-form solution fitting of the shape, with regularisation (prior towards the mean).
    
    [1] O. Aldrian & W. Smith, Inverse Rendering of Faces with a 3D Morphable Model, PAMI 2013.
    
    Note: Using less than the maximum number of coefficients to fit is not thoroughly tested yet and
    may contain an error.
    Note: Returns coefficients following standard normal distribution (i.e. all have similar magnitude).
    Why? Because we fit using the normalised basis?
    Note: The standard deviations given should be a vector, i.e. different for each landmark.
    This is not implemented yet.
    
    Args:
        shape_model:
            The Morphable Model whose shape (coefficients) are estimated.
        affine_camera_matrix:
            A 3x4 affine camera matrix from model to screen-space.
        landmarks:
            2D landmarks from an image to fit the model to.
        vertex_ids:
            The vertex ids in the model that correspond to the 2D points.
        base_face:
            The base or reference face from where the fitting is started. Usually this would be
            the models mean face, which is what will be used if the parameter is not explicitly specified.
        lambda_p:
            The regularisation parameter (weight of the prior towards the mean).
        num_coefficients_to_fit:
            How many shape-coefficients to fit (all others will stay 0). Should be
            bigger than zero, or std::nullopt to fit all coefficients.
        detector_standard_deviation:
            The standard deviation of the 2D landmarks given (e.g. of the detector used), in pixels.
        model_standard_deviation:
            The standard deviation of the 3D vertex points in the 3D model, projected
            to 2D (so the value is in pixels).
        
    Returns:
        The estimated shape-coefficients (alphas).
    """
    assert len(landmarks) == len(vertex_ids)
    
    num_coeffs_to_fit = num_coefficients_to_fit if num_coefficients_to_fit else\
        shape_model.get_num_principal_components()
    num_landmarks = len(landmarks)
    
    if not len(base_face):
        base_face = shape_model.mean
    
    # $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$
    # associated with the $N$ feature points.
    # And we insert a row of zeros after every third row, resulting in matrix
    # $\hat{V}_h \in R^{4N\times m-1}$:
    v_hat_h = np.zeros([4 * num_landmarks, num_coeffs_to_fit])
    # TODO: this could be implemented with better ways
    for i in range(num_landmarks):
        basis_rows = shape_model.get_rescaled_pca_basis_at_point(vertex_ids[i])
        v_hat_h[4 * i: 4 * i + 3, :] = basis_rows
    
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
    
    # The variances: Add the 2D and 3D standard deviations.
    # If the user doesn't provide them, we choose the following:
    # 2D (detector) standard deviation: In pixel, we follow [1] and choose sqrt(3) as the default value.
    # 3D (model) variance: 0.0f. It only makes sense to set it to something when we have a different
    # variance for different vertices.
    # The 3D variance has to be projected to 2D (for details, see paper [1]) so the units do match up.
    sigma_squared_2d = detector_standard_deviation ** 2 + model_standard_deviation ** 2
    omega = np.eye(3 * num_landmarks) / sigma_squared_2d
    
    # The landmarks in matrix notation (in homogeneous coordinates), $3N\times 1$
    y = np.ones([3 * num_landmarks])
    for i in range(num_landmarks):
        y[3 * i] = landmarks[i][0]
        y[3 * i + 1] = landmarks[i][1]
    
    # The mean, with an added homogeneous coordinate (x_1, y_1, z_1, 1, x_2, ...)^t
    v_bar = np.ones([4 * num_landmarks])
    for i in range(num_landmarks):
        v_bar[4 * i] = base_face[vertex_ids[i] * 3]
        v_bar[4 * i + 1] = base_face[vertex_ids[i] * 3 + 1]
        v_bar[4 * i + 2] = base_face[vertex_ids[i] * 3 + 2]
    
    # Bring into standard regularised quadratic form with diagonal distance matrix Omega:
    a = p.dot(v_hat_h)  # camera matrix times the basis
    b = p.dot(v_bar) - y  # camera matrix times the mean, minus the landmarks
    at_omega_reg = a.T.dot(omega).dot(a) + lambda_p * np.eye(num_coeffs_to_fit)
    # It's -A^t*Omega^t*b, but we don't need to transpose Omega, since it's a diagonal matrix.
    rhs = -a.T.dot(omega).dot(b)
    
    # c_s: The 'x' that we solve for. (The variance-normalised shape parameter vector, $c_s =
    # [a_1/sigma_{s,1} , ..., a_m-1/sigma_{s,m-1}]^t$.)
    # We get coefficients ~ N(0, 1), because we're fitting with the rescaled basis. The coefficients
    # are not multiplied with their eigenvalues.
    c_s = np.linalg.lstsq(at_omega_reg, rhs)[0]

    return c_s


def fit_fandp_to_landmarks_linear(shape_model, front_camera_matrix, front_landmarks, front_vertex_ids,
                                  profile_camera_matrix, profile_landmarks, profile_vertex_ids,
                                  base_face=list(), lambda_p=3.0, num_coefficients_to_fit=None,
                                  detector_standard_deviation=3.0 ** 0.5, model_standard_deviation=0.0):
    """
    Fits the shape of a Morphable Model to given 2D landmarks (i.e. estimates the maximum likelihood
    solution of the shape coefficients) as proposed in [1].
    It's a linear, closed-form solution fitting of the shape, with regularisation (prior towards the mean).

    [1] O. Aldrian & W. Smith, Inverse Rendering of Faces with a 3D Morphable Model, PAMI 2013.

    Note: Using less than the maximum number of coefficients to fit is not thoroughly tested yet and
    may contain an error.
    Note: Returns coefficients following standard normal distribution (i.e. all have similar magnitude).
    Why? Because we fit using the normalised basis?
    Note: The standard deviations given should be a vector, i.e. different for each landmark.
    This is not implemented yet.

    Args:
        shape_model:
            The Morphable Model whose shape (coefficients) are estimated.
        front_camera_matrix:
            A 3x4 affine camera matrix from model to screen-space of front picture.
        front_landmarks:
            2D landmarks from an image to fit the model to.
        front_vertex_ids:
            The vertex ids in the model that correspond to the 2D points.
        profile_camera_matrix:
            A 3x4 affine camera matrix from model to screen-space of profile picture.
        profile_landmarks:
            2D landmarks from an image to fit the model to.
        profile_vertex_ids:
            The vertex ids in the model that correspond to the 2D points.
        base_face:
            The base or reference face from where the fitting is started. Usually this would be
            the models mean face, which is what will be used if the parameter is not explicitly specified.
        lambda_p:
            The regularisation parameter (weight of the prior towards the mean).
        num_coefficients_to_fit:
            How many shape-coefficients to fit (all others will stay 0). Should be
            bigger than zero, or std::nullopt to fit all coefficients.
        detector_standard_deviation:
            The standard deviation of the 2D landmarks given (e.g. of the detector used), in pixels.
        model_standard_deviation:
            The standard deviation of the 3D vertex points in the 3D model, projected
            to 2D (so the value is in pixels).

    Returns:
        The estimated shape-coefficients (alphas).
    """
    assert len(front_landmarks) == len(front_vertex_ids)
    assert len(profile_landmarks) == len(profile_vertex_ids)
    front_landmarks = np.array(front_landmarks, copy=False)
    profile_landmarks = np.array(profile_landmarks, copy=False)

    num_coeffs_to_fit = num_coefficients_to_fit if num_coefficients_to_fit else \
        shape_model.get_num_principal_components()  # 特征向量的数量（就是需要拟合的参数数量）
    num_front_landmarks = len(front_landmarks)  # 正脸特征点的数量（0-based）
    num_profile_landmarks = len(profile_landmarks)  # 侧脸特征点的数量（0-based）
    num_total_landmarks = num_front_landmarks + num_profile_landmarks  # 总特征点的数量（0-based）

    if not len(base_face):
        base_face = shape_model.mean

    # $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$
    # associated with the $N$ feature points.
    # And we insert a row of zeros after every third row, resulting in matrix
    # $\hat{V}_h \in R^{4N\times m-1}$:
    v_hat_h = np.zeros([4 * num_total_landmarks, num_coeffs_to_fit])
    for i in range(num_front_landmarks):
        basis_rows = shape_model.get_rescaled_pca_basis_at_point(front_vertex_ids[i])
        v_hat_h[4 * i: 4 * i + 3, :] = basis_rows
    for i in range(num_front_landmarks, num_total_landmarks):
        basis_rows = shape_model.get_rescaled_pca_basis_at_point(profile_vertex_ids[i - num_front_landmarks])
        v_hat_h[4 * i: 4 * i + 3, :] = basis_rows

    # Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera matrix C
    # (P_Affine, affine_camera_matrix) is placed on the diagonal:
    p_row = []
    p_col = []
    p_data = []
    for i in range(num_front_landmarks):
        # 因为这里已知camera_matrix为3X4的矩阵，所以就不浪费时间做判断了。
        for x in range(3):
            for y in range(4):
                p_row.append(3 * i + x)
                p_col.append(4 * i + y)
                p_data.append(front_camera_matrix[x, y])
    for i in range(num_front_landmarks, num_total_landmarks):
        # 因为这里已知camera_matrix为3X4的矩阵，所以就不浪费时间做判断了。
        for x in range(3):
            for y in range(4):
                p_row.append(3 * i + x)
                p_col.append(4 * i + y)
                p_data.append(profile_camera_matrix[x, y])
    # 建立稀疏矩阵
    p = sparse.coo_matrix((p_data, (p_row, p_col)), shape=(3 * num_total_landmarks, 4 * num_total_landmarks))

    # The variances: Add the 2D and 3D standard deviations.
    # If the user doesn't provide them, we choose the following:
    # 2D (detector) standard deviation: In pixel, we follow [1] and choose sqrt(3) as the default value.
    # 3D (model) variance: 0.0f. It only makes sense to set it to something when we have a different
    # variance for different vertices.
    # The 3D variance has to be projected to 2D (for details, see paper [1]) so the units do match up.
    sigma_squared_2d = detector_standard_deviation ** 2 + model_standard_deviation ** 2
    omega = np.eye(3 * num_total_landmarks) / sigma_squared_2d

    # The landmarks in matrix notation (in homogeneous coordinates), $3N\times 1$
    y = np.ones([3 * num_total_landmarks])
    for i in range(num_front_landmarks):
        y[3 * i: 3 * i + 2] = front_landmarks[i]
    for i in range(num_front_landmarks, num_total_landmarks):
        y[3 * i: 3 * i + 2] = profile_landmarks[i - num_front_landmarks]

    # The mean, with an added homogeneous coordinate (x_1, y_1, z_1, 1, x_2, ...)^t
    v_bar = np.ones([4 * num_total_landmarks])
    for i in range(num_front_landmarks):
        v_bar[4 * i: 4 * i + 3] = base_face[front_vertex_ids[i] * 3: front_vertex_ids[i] * 3 + 3]
    for i in range(num_front_landmarks, num_total_landmarks):
        v_bar[4 * i: 4 * i + 3] = base_face[profile_vertex_ids[i - num_front_landmarks] * 3:
                                            profile_vertex_ids[i - num_front_landmarks] * 3 + 3]

    # Bring into standard regularised quadratic form with diagonal distance matrix Omega:
    a = p.dot(v_hat_h)  # camera matrix times the basis
    b = p.dot(v_bar) - y  # camera matrix times the mean, minus the landmarks
    # 经过正则化之后的等式左边的参数
    at_omega_reg = a.T.dot(omega).dot(a) + lambda_p * np.eye(num_coeffs_to_fit)
    # It's -A^t*Omega^t*b, but we don't need to transpose Omega, since it's a diagonal matrix.
    rhs = -a.T.dot(omega).dot(b)

    # c_s: The 'x' that we solve for. (The variance-normalised shape parameter vector, $c_s =
    # [a_1/sigma_{s,1} , ..., a_m-1/sigma_{s,m-1}]^t$.)
    # We get coefficients ~ N(0, 1), because we're fitting with the rescaled basis. The coefficients
    # are not multiplied with their eigenvalues.
    c_s = np.linalg.lstsq(at_omega_reg, rhs)[0]

    return c_s
