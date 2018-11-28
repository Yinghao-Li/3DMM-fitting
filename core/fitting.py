# -*- coding: utf-8 -*-
"""
@author: Yinghao Li
"""

from core import closest_edge_fitting, Blendshape, contour_correspondence, blendshape_fitting, EdgeTopology, \
    LandmarkMapper, linear_shape_fitting, glm, Landmark, MorphableModel, RenderingParameters, Mesh, \
    orthographic_camera_estimation_linear

import numpy as np
import copy
import math
import time
from typing import List, Tuple, Optional


def fit_shape(affine_camera_matrix, morphable_model, blendshapes, image_points, vertex_indices):
    """
    Convenience function that fits the shape model and expression blendshapes to
    landmarks. Makes the fitted PCA shape and blendshape coefficients accessible
    via the out parameters \p pca_shape_coefficients and \p blendshape_coefficients.
    It iterates PCA-shape and blendshape fitting until convergence
    (usually it converges within 5 to 10 iterations).
    
    See fit_shape_model(cv::Mat, eos::morphablemodel::MorphableModel, std::vector<eos::morphablemodel::Blendshape>,
                        std::vector<cv::Vec2f>, std::vector<int>, float lambda)
    for a simpler overload that just returns the shape instance.
    
    !! It should be noticed that this func is slightly different from the C++ version.
    !! pca_shape_coefficients and blendshape_coefficients are no longer parameters
    !! but returns.
    
    Args:
        affine_camera_matrix:
            The estimated pose as a 3x4 affine camera matrix that is used to fit the shape.
        morphable_model:
            The 3D Morphable Model used for the shape fitting.
        blendshapes:
            A vector of blendshapes that are being fit to the landmarks in addition to the PCA model.
        image_points:
            2D landmarks from an image to fit the model to.
        vertex_indices:
            The vertex indices in the model that correspond to the 2D points.
        
    Returns:
        combined_shape:
            The fitted model shape instance.
        pca_shape_coefficients:
            Output parameter that will contain the resulting pca shape coefficients.
        blendshape_coefficients:
            Output parameter that will contain the resulting blendshape coefficients.
    """
    blendshapes_as_basis = Blendshape.to_matrix(blendshapes)

    current_blendshape_coeffs = np.zeros(len(blendshapes))
    current_pca_coeffs = np.zeros(len(blendshapes))

    # run at least once:
    while True:
        last_blendshape_coeffs = current_blendshape_coeffs
        last_pca_coeffs = current_pca_coeffs
        # Estimate the PCA shape coefficients with the current blendshape coefficients (0 in the first iteration):
        mean_plus_blendshapes = morphable_model.shape_model.mean + blendshapes_as_basis.dot(last_blendshape_coeffs)
        current_pca_coeffs = linear_shape_fitting.\
            fit_shape_to_landmarks_linear(morphable_model.shape_model, affine_camera_matrix, image_points,
                                          vertex_indices, mean_plus_blendshapes, lambda_p=3.0,
                                          num_coefficients_to_fit=None)
        
        # Estimate the blendshape coefficients with the current PCA model estimate:
        pca_model_shape = morphable_model.shape_model.draw_sample(current_pca_coeffs)
        current_blendshape_coeffs = blendshape_fitting.\
            fit_blendshapes_to_landmarks_nnls(blendshapes, pca_model_shape, affine_camera_matrix,
                                              image_points, vertex_indices)

        if not (abs(np.linalg.norm(current_pca_coeffs) - np.linalg.norm(last_pca_coeffs)) >= 0.01 or
                abs(np.linalg.norm(current_blendshape_coeffs) - np.linalg.norm(last_blendshape_coeffs)) >= 0.01):
            break
        
    # Todo/Note: Could move next line outside the loop, not needed in here actually
    combined_shape = pca_model_shape + blendshapes_as_basis.dot(current_blendshape_coeffs)
    pca_shape_coefficients = current_pca_coeffs
    blendshape_coefficients = current_blendshape_coeffs
    
    return combined_shape, pca_shape_coefficients, blendshape_coefficients


def get_corresponding_pointset(landmarks, landmark_mapper, morphable_model):

    """
    Takes a LandmarkCollection of 2D landmarks and, using the landmark_mapper, finds the
    corresponding 3D vertex indices and returns them, along with the coordinates of the 3D points.
    
    The function only returns points which the landmark mapper was able to convert, and skips all
    points for which there is no mapping. Thus, the number of returned points might be smaller than
    the number of input points.
    All three output vectors have the same size and contain the points in the same order.
    \c landmarks can be an eos::core::LandmarkCollection<cv::Vec2f> or an rcr::LandmarkCollection<cv::Vec2f>.
    
    Notes:
        - Split into two functions, one which maps from 2D LMs to vtx_idx and returns a reduced vec of 2D LMs.
          And then the other one to go from vtx_idx to a vector<Vec4f>.
        - Place in a potentially more appropriate header (shape-fitting?).
        - Could move to detail namespace or forward-declare.
        - \c landmarks has to be a collection of LMs, with size(), [] and Vec2f ::coordinates.
        - Probably model_points would better be a Vector3f and not in homogeneous coordinates?
        
    Args:
        landmarks:
            A LandmarkCollection of 2D landmarks.
        landmark_mapper:
            A mapper which maps the 2D landmark identifiers to 3D model vertex indices.
        morphable_model:
            Model to get the 3D point coordinates from.
    
    Returns:
        A tuple of [image_points, model_points, vertex_indices].
    """
    # These will be the final 2D and 3D points used for the fitting:
    model_points = []  # the points in the 3D shape model
    vertex_indices = []  # their vertex indices
    image_points = []  # the corresponding 2D landmark points
    
    # Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
    for i in range(len(landmarks)):
        vertex_idx = landmark_mapper.convert(landmarks[i].name)
        # no mapping defined for the current landmark
        if not vertex_idx:
            continue
        vertex = morphable_model.shape_model.get_mean_at_point(vertex_idx)
        model_points.append(vertex + [1.0])
        vertex_indices.append(vertex_idx)
        image_points.append(landmarks[i].coordinates)
    return image_points, model_points, vertex_indices


def fit_shape_and_pose(morphable_model, blendshapes, landmarks, landmark_mapper, image_width, image_height,
                       edge_topology, contour_landmarks, model_contour, num_iterations=5,
                       num_shape_coefficients_to_fit=None, lambda_p=50.0, pca_shape_coefficients=None,
                       blendshape_coefficients=None, test_mode=False):
    """
    Fit the pose (camera), shape model, and expression blendshapes to landmarks in an iterative way.
    
    Convenience function that fits pose (camera), the shape model, and expression blendshapes
    to landmarks, in an iterative (alternating) way. It fits both sides of the face contour as well.
    
    If \p pca_shape_coefficients and/or \p blendshape_coefficients are given, they are used as
    starting values in the fitting. When the function returns, they contain the coefficients from
    the last iteration.
    
    \p num_iterations: Results are good for even a single iteration. For single-image fitting and
    for full convergence of all parameters, it can take up to 300 iterations. In tracking,
    particularly if initialising with the previous frame, it works well with as low as 1 to 5 iterations.
    \p edge_topology is used for the occluding-edge face contour fitting.
    \p contour_landmarks and \p model_contour are used to fit the front-facing contour.
    
    TODO: Add a convergence criterion.
    
    Args:
        morphable_model:
            The 3D Morphable Model used for the shape fitting.
        blendshapes:
            A vector of blendshapes that are being fit to the landmarks in addition to the PCA model.
        landmarks:
            2D landmarks from an image to fit the model to.
        landmark_mapper:
            Mapping info from the 2D landmark points to 3D vertex indices.
        image_width:
            Width of the input image (needed for the camera model).
        image_height:
            Height of the input image (needed for the camera model).
        edge_topology:
            Precomputed edge topology of the 3D model, needed for fast edge-lookup.
        contour_landmarks:
            2D image contour ids of left or right side (for example for ibug landmarks).
        model_contour:
            The model contour indices that should be considered to find the closest corresponding 3D vertex.
        num_iterations:
            Number of iterations that the different fitting parts will be alternated for.
        num_shape_coefficients_to_fit:
            How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or std::nullopt
            to fit all coefficients.
        lambda_p:
            Regularisation parameter of the PCA shape fitting.
        pca_shape_coefficients:
            If given, will be used as initial PCA shape coefficients to start the fitting. Will contain the final
            estimated coefficients.
        blendshape_coefficients:
            If given, will be used as initial expression blendshape coefficients to start the fitting.
            Will contain the final estimated coefficients.
        test_mode:
            Debug parameter: If true, the programme will print testing information.
            
    Returns:
        The fitted model shape instance and the final pose.
    """
    assert len(blendshapes) > 0
    assert len(landmarks) > 4
    assert image_height > 0 and image_width > 0
    assert num_iterations > 0
    
    # TODO: start time
    start_time = time.time()
    
    if not num_shape_coefficients_to_fit:
        num_shape_coefficients_to_fit = morphable_model.shape_model.get_num_principal_components()
    
    if not pca_shape_coefficients:
        pca_shape_coefficients = np.zeros([num_shape_coefficients_to_fit])
    # TODO: This leaves the following case open: num_coeffs given is empty or defined, but the
    # TODO: pca_shape_coefficients given is != num_coeffs or the model's max-coeffs. What to do then? Handle & document!
    
    if not blendshape_coefficients:
        blendshape_coefficients = np.zeros([len(blendshapes)])
    
    blendshapes_as_basis = Blendshape.to_matrix(blendshapes)
    
    # Current mesh - either from the given coefficients, or the mean:
    current_pca_shape = morphable_model.shape_model.draw_sample(pca_shape_coefficients)
    current_combined_shape = current_pca_shape + blendshapes_as_basis.dot(blendshape_coefficients)
    current_mesh = MorphableModel.sample_to_mesh(
        current_combined_shape, morphable_model.color_model.mean, morphable_model.shape_model.triangle_list,
        morphable_model.color_model.triangle_list, morphable_model.texture_coordinates)
    
    # TODO: time1
    if test_mode:
        print('time1: {}'.format(time.time() - start_time))
    
    # The 2D and 3D point correspondences used for the fitting:
    model_points = []  # the points in the 3D shape model
    vertex_indices = []  # their vertex indices
    image_points = []  # the corresponding 2D landmark points
    
    # Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM),
    # and get the corresponding model points (mean if given no initial coeffs, from the computed shape otherwise):
    for i in range(len(landmarks)):
        vertex_idx = landmark_mapper.convert(landmarks[i].name)
        # no mapping defined for the current landmark
        if not vertex_idx:
            continue
        vertex = np.hstack([current_mesh.vertices[vertex_idx], 1.0])
        model_points.append(vertex)
        vertex_indices.append(vertex_idx)
        image_points.append(landmarks[i].coordinates)
    
    # Need to do an initial pose fit to do the contour fitting inside the loop.
    # We'll do an expression fit too, since face shapes vary quite a lot, depending on expressions.
    current_pose = orthographic_camera_estimation_linear.estimate_orthographic_projection_linear(
        copy.deepcopy(image_points), copy.deepcopy(model_points), True, image_height)
    rendering_params = RenderingParameters.RenderingParameters(current_pose, image_width, image_height)
    
    affine_from_ortho = RenderingParameters.get_3x4_affine_camera_matrix(rendering_params, image_width, image_height)
    blendshape_coefficients = blendshape_fitting.fit_blendshapes_to_landmarks_nnls(
        blendshapes, current_pca_shape, affine_from_ortho, image_points, vertex_indices)
    
    # Mesh with same PCA coeffs as before, but new expression fit (this is relevant if no initial blendshape
    # coeffs have been given):
    current_combined_shape = current_pca_shape + Blendshape.to_matrix(blendshapes).dot(blendshape_coefficients)
    current_mesh = MorphableModel.sample_to_mesh(
        current_combined_shape, morphable_model.color_model.mean, morphable_model.shape_model.triangle_list,
        morphable_model.color_model.triangle_list, morphable_model.texture_coordinates)
    
    # The static (fixed) landmark correspondences which will stay the same throughout
    # the fitting (the inner face landmarks):
    fixed_image_points = copy.deepcopy(image_points)
    fixed_vertex_indices = copy.deepcopy(vertex_indices)
    
    # TODO: time2
    if test_mode:
        print('time2: {}'.format(time.time() - start_time))
    
    for i in range(num_iterations):
        
        # TODO: time2.1
        if test_mode:
            print('time2.1: {}'.format(time.time() - start_time))
        
        image_points = copy.deepcopy(fixed_image_points)
        vertex_indices = copy.deepcopy(fixed_vertex_indices)
        # Given the current pose, find 2D-3D contour correspondences of the front-facing face contour:
        yaw_angle = math.degrees(glm.yaw(rendering_params.get_rotation()))
        # For each 2D contour landmark, get the corresponding 3D vertex point and vertex id:
        image_points_contour, _, vertex_indices_contour = contour_correspondence.get_contour_correspondences(
            landmarks, contour_landmarks, model_contour, yaw_angle, current_mesh, rendering_params.get_modelview(),
            rendering_params.get_projection(), RenderingParameters.get_opencv_viewport(image_width, image_height))
        # Add the contour correspondences to the set of landmarks that we use for the fitting:
        vertex_indices += vertex_indices_contour
        image_points += image_points_contour
        
        # TODO: time2.2
        if test_mode:
            print('time2.2: {}'.format(time.time() - start_time))
        
        # Fit the occluding (away-facing) contour using the detected contour LMs:
        occluding_contour_landmarks = []
        # positive yaw = subject looking to the left
        if yaw_angle > 0.0:
            # the left contour is the occluding one we want to use ("away-facing")
            contour_landmarks_ = Landmark.landmark_filter(landmarks, contour_landmarks.left_contour)
            for lm in contour_landmarks_:
                occluding_contour_landmarks.append(lm.coordinates)
        else:
            contour_landmarks_ = Landmark.landmark_filter(landmarks, contour_landmarks.right_contour)
            for lm in contour_landmarks_:
                occluding_contour_landmarks.append(lm.coordinates)
                
        # TODO: time2.3
        if test_mode:
            print('time2.3: {}'.format(time.time() - start_time))
                
        edge_correspondences = closest_edge_fitting.find_occluding_edge_correspondences(
            current_mesh, edge_topology, rendering_params, occluding_contour_landmarks, 180.0)
        
        # TODO: time2.4
        if test_mode:
            print('time2.4: {}'.format(time.time() - start_time))
        
        image_points += edge_correspondences[0]
        vertex_indices += edge_correspondences[1]
        
        # TODO: time3
        if test_mode:
            print('time3: {}'.format(time.time() - start_time))
        
        # Get the model points of the current mesh, for all correspondences that we've got:
        model_points.clear()
        for v in vertex_indices:
            model_points.append(np.hstack([current_mesh.vertices[v], 1.0]))
            
        # Re-estimate the pose, using all correspondences:
        current_pose = orthographic_camera_estimation_linear.estimate_orthographic_projection_linear(
            copy.deepcopy(image_points), copy.deepcopy(model_points), True, image_height)
        rendering_params = RenderingParameters.RenderingParameters(current_pose, image_width, image_height)
        
        affine_from_ortho = RenderingParameters.get_3x4_affine_camera_matrix(
            rendering_params, image_width, image_height)
        
        # Estimate the PCA shape coefficients with the current blendshape coefficients:
        mean_plus_blendshapes = morphable_model.shape_model.mean + blendshapes_as_basis.dot(blendshape_coefficients)
        pca_shape_coefficients = linear_shape_fitting.fit_shape_to_landmarks_linear(
            morphable_model.shape_model, affine_from_ortho, image_points, vertex_indices,
            mean_plus_blendshapes, lambda_p, num_shape_coefficients_to_fit)
        
        # Estimate the blendshape coefficients with the current PCA model estimate:
        current_pca_shape = morphable_model.shape_model.draw_sample(pca_shape_coefficients)
        blendshape_coefficients = blendshape_fitting.fit_blendshapes_to_landmarks_nnls(
            blendshapes, current_pca_shape, affine_from_ortho, image_points, vertex_indices)
        
        current_combined_shape = current_pca_shape + blendshapes_as_basis.dot(blendshape_coefficients)
        current_mesh = MorphableModel.sample_to_mesh(
            current_combined_shape, morphable_model.color_model.mean, morphable_model.shape_model.triangle_list,
            morphable_model.color_model.triangle_list, morphable_model.texture_coordinates)
        
        # TODO: time4
        if test_mode:
            print('time4: {}'.format(time.time() - start_time))
    
    # fitted_image_points = image_points
    return current_mesh, rendering_params
            
        
def fit_profile(morphable_model, blendshapes, profile_landmarks, profile_mapper, image_width, image_height,
                num_iterations=5, num_shape_coefficients_to_fit=None,
                lambda_p=50.0, pca_shape_coefficients=None, blendshape_coefficients=None,
                test_mode=False):
    """
    Fit the pose (camera), shape model, and expression blendshapes to landmarks in an iterative way.

    Convenience function that fits pose (camera), the shape model, and expression blendshapes
    to landmarks, in an iterative (alternating) way. It fits both sides of the face contour as well.

    If \p pca_shape_coefficients and/or \p blendshape_coefficients are given, they are used as
    starting values in the fitting. When the function returns, they contain the coefficients from
    the last iteration.

    \p num_iterations: Results are good for even a single iteration. For single-image fitting and
    for full convergence of all parameters, it can take up to 300 iterations. In tracking,
    particularly if initialising with the previous frame, it works well with as low as 1 to 5 iterations.
    \p edge_topology is used for the occluding-edge face contour fitting.
    \p contour_landmarks and \p model_contour are used to fit the front-facing contour.

    TODO: Add a convergence criterion.

    Args:
        morphable_model:
            The 3D Morphable Model used for the shape fitting.
        blendshapes:
            A vector of blendshapes that are being fit to the landmarks in addition to the PCA model.
        profile_landmarks:
            2D profile landmarks from a profile to fit the model to.
        profile_mapper:
            Mapping info from the profile landmark indices to 3D vertex indices.
        image_width:
            Width of the input image (needed for the camera model).
        image_height:
            Height of the input image (needed for the camera model).
        num_iterations:
            Number of iterations that the different fitting parts will be alternated for.
        num_shape_coefficients_to_fit:
            How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or std::nullopt to
            fit all coefficients.
        lambda_p:
            Regularisation parameter of the PCA shape fitting.
        pca_shape_coefficients:
            If given, will be used as initial PCA shape coefficients to start the fitting. Will contain the final
            estimated coefficients.
        blendshape_coefficients:
            If given, will be used as initial expression blendshape coefficients to start the fitting.
            Will contain the final estimated coefficients.
        test_mode:
            Debug parameter: If true, the programme will print testing information.

    Returns:
        The fitted model shape instance and the final pose.
    """
    assert len(blendshapes) > 0
    assert len(profile_landmarks) > 4
    assert image_height > 0 and image_width > 0
    assert num_iterations > 0
    assert len(pca_shape_coefficients) <= morphable_model.shape_model.get_num_principal_components()

    # TODO: start time
    start_time = time.time()

    if not num_shape_coefficients_to_fit:
        num_shape_coefficients_to_fit = morphable_model.shape_model.get_num_principal_components()

    if not pca_shape_coefficients:
        pca_shape_coefficients = np.zeros([num_shape_coefficients_to_fit])
    # TODO: This leaves the following case open: num_coeffs given is empty or defined, but the
    # pca_shape_coefficients given is != num_coeffs or the model's max-coeffs. What to do then? Handle & document!

    if not blendshape_coefficients:
        blendshape_coefficients = np.zeros([len(blendshapes)])

    blendshapes_as_basis = Blendshape.to_matrix(blendshapes)

    # Current mesh - either from the given coefficients, or the mean:
    current_pca_shape = morphable_model.shape_model.draw_sample(pca_shape_coefficients)
    current_combined_shape = current_pca_shape + blendshapes_as_basis.dot(blendshape_coefficients)
    current_mesh = MorphableModel.sample_to_mesh(
        current_combined_shape, morphable_model.color_model.mean, morphable_model.shape_model.triangle_list,
        morphable_model.color_model.triangle_list, morphable_model.texture_coordinates)

    # The 2D and 3D point correspondences used for the fitting:
    model_points = []  # the points in the 3D shape model
    vertex_indices = []  # their vertex indices
    image_points = []  # the corresponding 2D landmark points

    # Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM),
    # and get the corresponding model points (mean if given no initial coeffs, from the computed shape otherwise):
    for i in range(len(profile_landmarks)):
        # TODO: 这里先用右侧脸的特征点做测试，最后应该加入自动判断左右脸的特征点的代码。
        vertex_idx = profile_mapper.right_mapper[profile_landmarks[i].name]
        # no mapping defined for the current landmark
        if not vertex_idx:
            continue
        vertex = np.hstack([current_mesh.vertices[vertex_idx], 1.0])
        model_points.append(vertex)
        vertex_indices.append(vertex_idx)
        image_points.append(profile_landmarks[i].coordinates)

    # Need to do an initial pose fit to do the contour fitting inside the loop.
    # We'll do an expression fit too, since face shapes vary quite a lot, depending on expressions.
    current_pose = orthographic_camera_estimation_linear.estimate_orthographic_projection_linear(
        copy.deepcopy(image_points), copy.deepcopy(model_points), True, image_height)
    rendering_params = RenderingParameters.RenderingParameters(current_pose, image_width, image_height)

    affine_from_ortho = RenderingParameters.get_3x4_affine_camera_matrix(rendering_params, image_width, image_height)
    blendshape_coefficients = blendshape_fitting.fit_blendshapes_to_landmarks_nnls(
        blendshapes, current_pca_shape, affine_from_ortho, image_points, vertex_indices)

    # Mesh with same PCA coeffs as before, but new expression fit
    # (this is relevant if no initial blendshape coeffs have been given):
    current_combined_shape = current_pca_shape + Blendshape.to_matrix(blendshapes).dot(blendshape_coefficients)
    current_mesh = MorphableModel.sample_to_mesh(
        current_combined_shape, morphable_model.color_model.mean, morphable_model.shape_model.triangle_list,
        morphable_model.color_model.triangle_list, morphable_model.texture_coordinates)

    # The static (fixed) landmark correspondences which will stay the same throughout
    # the fitting (the inner face landmarks):
    fixed_image_points = copy.deepcopy(image_points)
    fixed_vertex_indices = copy.deepcopy(vertex_indices)

    for i in range(num_iterations):

        image_points = copy.deepcopy(fixed_image_points)
        vertex_indices = copy.deepcopy(fixed_vertex_indices)

        # Get the model points of the current mesh, for all correspondences that we've got:
        model_points.clear()
        for v in vertex_indices:
            model_points.append(np.hstack([current_mesh.vertices[v], 1.0]))

        # Re-estimate the pose, using all correspondences:
        current_pose = orthographic_camera_estimation_linear. estimate_orthographic_projection_linear(
            copy.deepcopy(image_points), copy.deepcopy(model_points), True, image_height)
        rendering_params = RenderingParameters.RenderingParameters(current_pose, image_width, image_height)

        affine_from_ortho = RenderingParameters.get_3x4_affine_camera_matrix(
            rendering_params, image_width, image_height)

        # Estimate the PCA shape coefficients with the current blendshape coefficients:
        mean_plus_blendshapes = morphable_model.shape_model.mean + blendshapes_as_basis.dot(blendshape_coefficients)
        pca_shape_coefficients = linear_shape_fitting.fit_shape_to_landmarks_linear(
            morphable_model.shape_model, affine_from_ortho, image_points, vertex_indices,
            mean_plus_blendshapes, lambda_p, num_shape_coefficients_to_fit)

        # Estimate the blendshape coefficients with the current PCA model estimate:
        current_pca_shape = morphable_model.shape_model.draw_sample(pca_shape_coefficients)
        blendshape_coefficients = blendshape_fitting.fit_blendshapes_to_landmarks_nnls(
            blendshapes, current_pca_shape, affine_from_ortho, image_points, vertex_indices)

        current_combined_shape = current_pca_shape + blendshapes_as_basis.dot(blendshape_coefficients)
        current_mesh = MorphableModel.sample_to_mesh(
            current_combined_shape, morphable_model.color_model.mean, morphable_model.shape_model.triangle_list,
            morphable_model.color_model.triangle_list, morphable_model.texture_coordinates)

        # TODO: time4
        if test_mode:
            print('time4: {}'.format(time.time() - start_time))

    # fitted_image_points = image_points
    return current_mesh, rendering_params, blendshape_coefficients


def fit_front_and_profile(
        morphable_model: MorphableModel.MorphableModel, blendshapes: List[Blendshape.Blendshape],
        front_landmarks: List[Landmark.Landmark], front_landmark_mapper: LandmarkMapper.LandmarkMapper,
        profile_landmarks: List[Landmark.Landmark], profile_landmark_mapper: LandmarkMapper.ProfileLandmarkMapper,
        image_width: int, image_height: int, edge_topology: EdgeTopology.EdgeTopology,
        contour_landmarks: contour_correspondence.ContourLandmarks, model_contour: contour_correspondence.ModelContour,
        num_iterations: Optional[int] = 5, num_shape_coefficients_to_fit: Optional[int] = None,
        lambda_p: Optional[float] = 10.0, pca_shape_coefficients: Optional[list] = list(),
        blendshape_coefficients: Optional[list] = list())\
        -> Tuple[Mesh.Mesh, RenderingParameters.RenderingParameters, RenderingParameters.RenderingParameters]:
    """
    Fit the pose (camera), shape model, and expression blendshapes to landmarks in an iterative way.

    Convenience function that fits pose (camera), the shape model, and expression blendshapes
    to landmarks, in an iterative (alternating) way. It fits both sides of the face contour as well.

    If pca_shape_coefficients and/or blendshape_coefficients are given, they are used as
    starting values in the fitting. When the function returns, they contain the coefficients from
    the last iteration.

    num_iterations: Results are good for even a single iteration. For single-image fitting and
    for full convergence of all parameters, it can take up to 300 iterations. In tracking,
    particularly if initialising with the previous frame, it works well with as low as 1 to 5 iterations.
    edge_topology is used for the occluding-edge face contour fitting.
    contour_landmarks and model_contour are used to fit the front-facing contour.

    Args:
        morphable_model:
            The 3D Morphable Model used for the shape fitting.
        blendshapes:
            A vector of blendshapes that are being fit to the landmarks in addition to the PCA model.
        front_landmarks:
            2D landmarks from a front image to fit the model to.
        front_landmark_mapper:
            Mapping info from the front 2D landmark points to 3D vertex indices.
        profile_landmarks:
            2D landmarks from a profile image to fit the model to.
        profile_landmark_mapper:
            Mapping info from the profile 2D landmark points to 3D vertex indices.
        image_width:
            Width of the input image (needed for the camera model).
            这里假设正脸与侧脸图片的宽度与高度相同。
        image_height:
            Height of the input image (needed for the camera model).
            这里假设正脸与侧脸图片的宽度与高度相同。
        edge_topology:
            Precomputed edge topology of the 3D model, needed for fast edge-lookup.
        contour_landmarks:
            2D image contour ids of left or right side (for example for ibug landmarks).
        model_contour:
            The model contour indices that should be considered to find the closest corresponding 3D vertex.
        num_iterations:
            Number of iterations that the different fitting parts will be alternated for.
        num_shape_coefficients_to_fit:
            How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or std::nullopt
            to fit all coefficients.
        lambda_p:
            Regularisation parameter of the PCA shape fitting.
        pca_shape_coefficients:
            If given, will be used as initial PCA shape coefficients to start the fitting. Will contain the final
            estimated coefficients.
        blendshape_coefficients:
            If given, will be used as initial expression blendshape coefficients to start the fitting.
            Will contain the final estimated coefficients.

    Returns:
        The fitted model shape instance and the final pose.
    """
    assert len(blendshapes) > 0
    assert len(front_landmarks) > 4
    assert image_height > 0 and image_width > 0
    assert num_iterations > 0

    # TODO: start time
    start_time = time.time()

    if not num_shape_coefficients_to_fit:
        num_shape_coefficients_to_fit = morphable_model.shape_model.get_num_principal_components()

    if not len(pca_shape_coefficients):
        pca_shape_coefficients = np.zeros([num_shape_coefficients_to_fit])

    if not len(blendshape_coefficients):
        blendshape_coefficients = np.zeros([len(blendshapes)])

    blendshapes_as_basis = Blendshape.to_matrix(blendshapes)

    # Current mesh - either from the given coefficients, or the mean:
    # Current mesh在这里计算一次即可，因为它没有用到图片的特征点，即对正脸与侧脸图片，它的输出一致。
    current_pca_shape = morphable_model.shape_model.draw_sample(pca_shape_coefficients)
    current_combined_shape = current_pca_shape + blendshapes_as_basis.dot(blendshape_coefficients)
    current_mesh = MorphableModel.sample_to_mesh(
        current_combined_shape, morphable_model.color_model.mean, morphable_model.shape_model.triangle_list,
        morphable_model.color_model.triangle_list, morphable_model.texture_coordinates)

    # The 2D and 3D point correspondences used for the fitting:
    # 正脸与侧脸特征点在这里开始区分。
    front_model_points = []  # the points in the 3D shape model for frontal face
    front_vertex_indices = []  # their vertex indices for frontal face
    front_image_points = []  # the corresponding 2D landmark points for frontal face
    profile_model_points = []  # the points in the 3D shape model for profile face
    profile_vertex_indices = []  # their vertex indices for profile face
    profile_image_points = []  # the corresponding 2D landmark points for profile face

    # Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM),
    # and get the corresponding model points (mean if given no initial coeffs, from the computed shape otherwise):
    for i in range(len(front_landmarks)):
        vertex_idx = front_landmark_mapper.convert(front_landmarks[i].name)
        # no mapping defined for the current landmark
        if not vertex_idx:
            continue
        vertex = np.hstack([current_mesh.vertices[vertex_idx], 1.0])
        front_model_points.append(vertex)
        front_vertex_indices.append(vertex_idx)
        front_image_points.append(front_landmarks[i].coordinates)

    # 对侧脸特征点重复上面的步骤。
    for i in range(len(profile_landmarks)):
        vertex_idx = profile_landmark_mapper.right_mapper[profile_landmarks[i].name]
        # no mapping defined for the current landmark
        if not vertex_idx:
            continue
        vertex = np.hstack([current_mesh.vertices[vertex_idx], 1.0])
        profile_model_points.append(vertex)
        profile_vertex_indices.append(vertex_idx)
        profile_image_points.append(profile_landmarks[i].coordinates)

    # Need to do an initial pose fit to do the contour fitting inside the loop.
    # We'll do an expression fit too, since face shapes vary quite a lot, depending on expressions.
    # 正脸特征点的参数拟合。
    current_front_pose = orthographic_camera_estimation_linear.estimate_orthographic_projection_linear(
        copy.deepcopy(front_image_points), copy.deepcopy(front_model_points), True, image_height)
    front_rendering_params = RenderingParameters.RenderingParameters(current_front_pose, image_width, image_height)
    affine_from_ortho = RenderingParameters.get_3x4_affine_camera_matrix(
        front_rendering_params, image_width, image_height)
    blendshape_coefficients = blendshape_fitting.fit_blendshapes_to_landmarks_nnls(
        blendshapes, current_pca_shape, affine_from_ortho, front_image_points, front_vertex_indices)

    current_profile_pose = orthographic_camera_estimation_linear.estimate_orthographic_projection_linear(
        copy.deepcopy(profile_image_points), copy.deepcopy(profile_model_points), True, image_height)
    profile_rendering_params = RenderingParameters.RenderingParameters(
        current_profile_pose, image_width, image_height)

    # Mesh with same PCA coeffs as before, but new expression fit (this is relevant if no initial blendshape
    # coeffs have been given):
    current_combined_shape = current_pca_shape + blendshapes_as_basis.dot(blendshape_coefficients)
    current_mesh = MorphableModel.sample_to_mesh(
        current_combined_shape, morphable_model.color_model.mean, morphable_model.shape_model.triangle_list,
        morphable_model.color_model.triangle_list, morphable_model.texture_coordinates)

    # The static (fixed) landmark correspondences which will stay the same throughout
    # the fitting (the inner face landmarks):
    fixed_front_image_points = copy.deepcopy(front_image_points)
    fixed_front_vertex_indices = copy.deepcopy(front_vertex_indices)

    pca_shape_coefficients = np.zeros(np.shape(morphable_model.shape_model.get_num_principal_components()))
    for i in range(num_iterations):
        # TODO: to delete
        blendshapes_as_basis = np.zeros(np.shape(blendshapes_as_basis))

        front_image_points = copy.deepcopy(fixed_front_image_points)
        front_vertex_indices = copy.deepcopy(fixed_front_vertex_indices)

        # Given the current pose, find 2D-3D contour correspondences of the front-facing face contour:
        yaw_angle = math.degrees(glm.yaw(front_rendering_params.get_rotation()))
        print('profile-yaw: {}'.format(math.degrees(glm.yaw(profile_rendering_params.get_rotation()))))
        print('profile-pitch: {}'.format(math.degrees(glm.pitch(profile_rendering_params.get_rotation()))))
        print('profile-roll: {}'.format(math.degrees(glm.roll(profile_rendering_params.get_rotation()))))
        # For each 2D contour landmark, get the corresponding 3D vertex point and vertex id:
        image_points_contour, _, vertex_indices_contour = contour_correspondence.get_contour_correspondences(
            front_landmarks, contour_landmarks, model_contour, yaw_angle, current_mesh,
            front_rendering_params.get_modelview(), front_rendering_params.get_projection(),
            RenderingParameters.get_opencv_viewport(image_width, image_height))
        # Add the contour correspondences to the set of landmarks that we use for the fitting:
        front_vertex_indices += vertex_indices_contour
        front_image_points += image_points_contour

        # Fit the occluding (away-facing) contour using the detected contour LMs:
        occluding_contour_landmarks = []
        # positive yaw = subject looking to the left
        if yaw_angle > 0.0:
            # the left contour is the occluding one we want to use ("away-facing")
            contour_landmarks_ = Landmark.landmark_filter(front_landmarks, contour_landmarks.left_contour)
            for lm in contour_landmarks_:
                occluding_contour_landmarks.append(lm.coordinates)
        else:
            contour_landmarks_ = Landmark.landmark_filter(front_landmarks, contour_landmarks.right_contour)
            for lm in contour_landmarks_:
                occluding_contour_landmarks.append(lm.coordinates)

        if i < 1:
            edge_correspondences = closest_edge_fitting.find_occluding_edge_correspondences(
                current_mesh, edge_topology, front_rendering_params, occluding_contour_landmarks)

        front_image_points += edge_correspondences[0]
        front_vertex_indices += edge_correspondences[1]

        # Get the model points of the current mesh, for all correspondences that we've got:
        front_model_points.clear()
        for v in front_vertex_indices:
            front_model_points.append(np.hstack([current_mesh.vertices[v], 1.0]))

        profile_model_points.clear()
        for v in profile_vertex_indices:
            profile_model_points.append(np.hstack([current_mesh.vertices[v], 1.0]))

        # Re-estimate the pose, using all correspondences:
        # 对正脸特征点拟合
        current_front_pose = orthographic_camera_estimation_linear.estimate_orthographic_projection_linear(
            copy.deepcopy(front_image_points), copy.deepcopy(front_model_points), True, image_height)
        front_rendering_params = RenderingParameters.RenderingParameters(current_front_pose, image_width, image_height)
        front_affine_from_ortho = RenderingParameters.get_3x4_affine_camera_matrix(
            front_rendering_params, image_width, image_height)

        # 对侧脸特征点拟合
        if i < 3:
            current_profile_pose = orthographic_camera_estimation_linear.estimate_orthographic_projection_linear(
                copy.deepcopy(profile_image_points), copy.deepcopy(profile_model_points), True, image_height)
            profile_rendering_params = RenderingParameters.RenderingParameters(
                current_profile_pose, image_width, image_height)
            profile_affine_from_ortho = RenderingParameters.get_3x4_affine_camera_matrix(
                profile_rendering_params, image_width, image_height)

        pre_shape_coefficients = pca_shape_coefficients.copy()

        # Estimate the PCA shape coefficients with the current blendshape coefficients:
        mean_plus_blendshapes = morphable_model.shape_model.mean + blendshapes_as_basis.dot(blendshape_coefficients)
        pca_shape_coefficients = linear_shape_fitting.fit_fandp_to_landmarks_linear(
            morphable_model.shape_model, front_affine_from_ortho, front_image_points, front_vertex_indices,
            profile_affine_from_ortho, profile_image_points, profile_vertex_indices, mean_plus_blendshapes,
            lambda_p, num_shape_coefficients_to_fit)

        # Estimate the blendshape coefficients with the current PCA model estimate:
        current_pca_shape = morphable_model.shape_model.draw_sample(pca_shape_coefficients)
        blendshape_coefficients = blendshape_fitting.fit_blendshapes_to_landmarks_nnls(
            blendshapes, current_pca_shape, affine_from_ortho, front_image_points, front_vertex_indices)

        current_combined_shape = current_pca_shape + blendshapes_as_basis.dot(blendshape_coefficients)
        current_mesh = MorphableModel.sample_to_mesh(
            current_combined_shape, morphable_model.color_model.mean, morphable_model.shape_model.triangle_list,
            morphable_model.color_model.triangle_list, morphable_model.texture_coordinates)

        print(np.linalg.norm(pca_shape_coefficients - pre_shape_coefficients))
        if np.linalg.norm(pca_shape_coefficients - pre_shape_coefficients) < 1e-3:
            break

        print('loop {} is finished at time: {}'.format(i + 1, time.time() - start_time))

    return current_mesh, front_rendering_params, profile_rendering_params


def fit_front_calc_profile(
        morphable_model: MorphableModel.MorphableModel, blendshapes: List[Blendshape.Blendshape],
        front_landmarks: List[Landmark.Landmark], front_landmark_mapper: LandmarkMapper.LandmarkMapper,
        profile_landmarks: List[Landmark.Landmark], profile_landmark_mapper: LandmarkMapper.ProfileLandmarkMapper,
        image_width: int, image_height: int, edge_topology: EdgeTopology.EdgeTopology,
        contour_landmarks: contour_correspondence.ContourLandmarks, model_contour: contour_correspondence.ModelContour,
        num_iterations: Optional[int] = 5, num_shape_coefficients_to_fit: Optional[int] = None,
        lambda_p: Optional[float] = 10.0, pca_shape_coefficients: Optional[list] = list(),
        blendshape_coefficients: Optional[list] = list())\
        -> Tuple[Mesh.Mesh, RenderingParameters.RenderingParameters, RenderingParameters.RenderingParameters]:
    """
    Fit the pose (camera), shape model, and expression blendshapes to landmarks in an iterative way.

    Convenience function that fits pose (camera), the shape model, and expression blendshapes
    to landmarks, in an iterative (alternating) way. It fits both sides of the face contour as well.

    If pca_shape_coefficients and/or blendshape_coefficients are given, they are used as
    starting values in the fitting. When the function returns, they contain the coefficients from
    the last iteration.

    num_iterations: Results are good for even a single iteration. For single-image fitting and
    for full convergence of all parameters, it can take up to 300 iterations. In tracking,
    particularly if initialising with the previous frame, it works well with as low as 1 to 5 iterations.
    edge_topology is used for the occluding-edge face contour fitting.
    contour_landmarks and model_contour are used to fit the front-facing contour.

    TODO: Add a convergence criterion.

    Args:
        morphable_model:
            The 3D Morphable Model used for the shape fitting.
        blendshapes:
            A vector of blendshapes that are being fit to the landmarks in addition to the PCA model.
        front_landmarks:
            2D landmarks from a front image to fit the model to.
        front_landmark_mapper:
            Mapping info from the front 2D landmark points to 3D vertex indices.
        profile_landmarks:
            2D landmarks from a profile image to fit the model to.
        profile_landmark_mapper:
            Mapping info from the profile 2D landmark points to 3D vertex indices.
        image_width:
            Width of the input image (needed for the camera model).
            这里假设正脸与侧脸图片的宽度与高度相同。
        image_height:
            Height of the input image (needed for the camera model).
            这里假设正脸与侧脸图片的宽度与高度相同。
        edge_topology:
            Precomputed edge topology of the 3D model, needed for fast edge-lookup.
        contour_landmarks:
            2D image contour ids of left or right side (for example for ibug landmarks).
        model_contour:
            The model contour indices that should be considered to find the closest corresponding 3D vertex.
        num_iterations:
            Number of iterations that the different fitting parts will be alternated for.
        num_shape_coefficients_to_fit:
            How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or std::nullopt
            to fit all coefficients.
        lambda_p:
            Regularisation parameter of the PCA shape fitting.
        pca_shape_coefficients:
            If given, will be used as initial PCA shape coefficients to start the fitting. Will contain the final
            estimated coefficients.
        blendshape_coefficients:
            If given, will be used as initial expression blendshape coefficients to start the fitting.
            Will contain the final estimated coefficients.

    Returns:
        The fitted model shape instance and the final pose.
    """
    assert len(blendshapes) > 0
    assert len(front_landmarks) > 4
    assert image_height > 0 and image_width > 0
    assert num_iterations > 0

    # TODO: start time
    start_time = time.time()

    if not num_shape_coefficients_to_fit:
        num_shape_coefficients_to_fit = morphable_model.shape_model.get_num_principal_components()

    if not len(pca_shape_coefficients):
        pca_shape_coefficients = np.zeros([num_shape_coefficients_to_fit])

    if not len(blendshape_coefficients):
        blendshape_coefficients = np.zeros([len(blendshapes)])

    blendshapes_as_basis = Blendshape.to_matrix(blendshapes)

    # Current mesh - either from the given coefficients, or the mean:
    # Current mesh在这里计算一次即可，因为它没有用到图片的特征点，即对正脸与侧脸图片，它的输出一致。
    current_pca_shape = morphable_model.shape_model.draw_sample(pca_shape_coefficients)
    current_combined_shape = current_pca_shape + blendshapes_as_basis.dot(blendshape_coefficients)
    current_mesh = MorphableModel.sample_to_mesh(
        current_combined_shape, morphable_model.color_model.mean, morphable_model.shape_model.triangle_list,
        morphable_model.color_model.triangle_list, morphable_model.texture_coordinates)

    # The 2D and 3D point correspondences used for the fitting:
    # 正脸与侧脸特征点在这里开始区分。
    front_model_points = []  # the points in the 3D shape model for frontal face
    front_vertex_indices = []  # their vertex indices for frontal face
    front_image_points = []  # the corresponding 2D landmark points for frontal face
    profile_model_points = []  # the points in the 3D shape model for profile face
    profile_vertex_indices = []  # their vertex indices for profile face
    profile_image_points = []  # the corresponding 2D landmark points for profile face

    # Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM),
    # and get the corresponding model points (mean if given no initial coeffs, from the computed shape otherwise):
    for i in range(len(front_landmarks)):
        vertex_idx = front_landmark_mapper.convert(front_landmarks[i].name)
        # no mapping defined for the current landmark
        if not vertex_idx:
            continue
        vertex = np.hstack([current_mesh.vertices[vertex_idx], 1.0])
        front_model_points.append(vertex)
        front_vertex_indices.append(vertex_idx)
        front_image_points.append(front_landmarks[i].coordinates)

    # 对侧脸特征点重复上面的步骤。
    for i in range(len(profile_landmarks)):
        vertex_idx = profile_landmark_mapper.right_mapper[profile_landmarks[i].name]
        # no mapping defined for the current landmark
        if not vertex_idx:
            continue
        vertex = np.hstack([current_mesh.vertices[vertex_idx], 1.0])
        profile_model_points.append(vertex)
        profile_vertex_indices.append(vertex_idx)
        profile_image_points.append(profile_landmarks[i].coordinates)

    # Need to do an initial pose fit to do the contour fitting inside the loop.
    # We'll do an expression fit too, since face shapes vary quite a lot, depending on expressions.
    # 正脸特征点的参数拟合。
    current_front_pose = orthographic_camera_estimation_linear.estimate_orthographic_projection_linear(
        copy.deepcopy(front_image_points), copy.deepcopy(front_model_points), True, image_height)
    front_rendering_params = RenderingParameters.RenderingParameters(current_front_pose, image_width, image_height)
    affine_from_ortho = RenderingParameters.get_3x4_affine_camera_matrix(
        front_rendering_params, image_width, image_height)
    blendshape_coefficients = blendshape_fitting.fit_blendshapes_to_landmarks_nnls(
        blendshapes, current_pca_shape, affine_from_ortho, front_image_points, front_vertex_indices)

    current_profile_pose = orthographic_camera_estimation_linear.estimate_orthographic_projection_linear(
        copy.deepcopy(profile_image_points), copy.deepcopy(profile_model_points), True, image_height)
    profile_rendering_params = RenderingParameters.RenderingParameters(
        current_profile_pose, image_width, image_height)

    # Mesh with same PCA coeffs as before, but new expression fit (this is relevant if no initial blendshape
    # coeffs have been given):
    current_combined_shape = current_pca_shape + blendshapes_as_basis.dot(blendshape_coefficients)
    current_mesh = MorphableModel.sample_to_mesh(
        current_combined_shape, morphable_model.color_model.mean, morphable_model.shape_model.triangle_list,
        morphable_model.color_model.triangle_list, morphable_model.texture_coordinates)

    # The static (fixed) landmark correspondences which will stay the same throughout
    # the fitting (the inner face landmarks):
    fixed_front_image_points = copy.deepcopy(front_image_points)
    fixed_front_vertex_indices = copy.deepcopy(front_vertex_indices)

    pca_shape_coefficients = np.zeros(np.shape(morphable_model.shape_model.get_num_principal_components()))
    for i in range(num_iterations):
        # TODO: to delete
        blendshapes_as_basis = np.zeros(np.shape(blendshapes_as_basis))

        front_image_points = copy.deepcopy(fixed_front_image_points)
        front_vertex_indices = copy.deepcopy(fixed_front_vertex_indices)

        # Given the current pose, find 2D-3D contour correspondences of the front-facing face contour:
        yaw_angle = math.degrees(glm.yaw(front_rendering_params.get_rotation()))
        # For each 2D contour landmark, get the corresponding 3D vertex point and vertex id:
        image_points_contour, _, vertex_indices_contour = contour_correspondence.get_contour_correspondences(
            front_landmarks, contour_landmarks, model_contour, yaw_angle, current_mesh,
            front_rendering_params.get_modelview(), front_rendering_params.get_projection(),
            RenderingParameters.get_opencv_viewport(image_width, image_height))
        # Add the contour correspondences to the set of landmarks that we use for the fitting:
        front_vertex_indices += vertex_indices_contour
        front_image_points += image_points_contour

        # Fit the occluding (away-facing) contour using the detected contour LMs:
        occluding_contour_landmarks = []
        # positive yaw = subject looking to the left
        if yaw_angle > 0.0:
            # the left contour is the occluding one we want to use ("away-facing")
            contour_landmarks_ = Landmark.landmark_filter(front_landmarks, contour_landmarks.left_contour)
            for lm in contour_landmarks_:
                occluding_contour_landmarks.append(lm.coordinates)
        else:
            contour_landmarks_ = Landmark.landmark_filter(front_landmarks, contour_landmarks.right_contour)
            for lm in contour_landmarks_:
                occluding_contour_landmarks.append(lm.coordinates)

        if i < 3:
            edge_correspondences = closest_edge_fitting.find_occluding_edge_correspondences(
                current_mesh, edge_topology, front_rendering_params, occluding_contour_landmarks)

        front_image_points += edge_correspondences[0]
        front_vertex_indices += edge_correspondences[1]

        # Get the model points of the current mesh, for all correspondences that we've got:
        front_model_points.clear()
        for v in front_vertex_indices:
            front_model_points.append(np.hstack([current_mesh.vertices[v], 1.0]))

        # Re-estimate the pose, using all correspondences:
        # 对正脸特征点拟合
        current_front_pose = orthographic_camera_estimation_linear.estimate_orthographic_projection_linear(
            copy.deepcopy(front_image_points), copy.deepcopy(front_model_points), True, image_height)
        front_rendering_params = RenderingParameters.RenderingParameters(current_front_pose, image_width, image_height)
        front_affine_from_ortho = RenderingParameters.get_3x4_affine_camera_matrix(
            front_rendering_params, image_width, image_height)

        pre_shape_coefficients = pca_shape_coefficients.copy()

        # Estimate the PCA shape coefficients with the current blendshape coefficients:
        mean_plus_blendshapes = morphable_model.shape_model.mean + blendshapes_as_basis.dot(blendshape_coefficients)
        pca_shape_coefficients = linear_shape_fitting.fit_shape_to_landmarks_linear(
            morphable_model.shape_model, front_affine_from_ortho, front_image_points, front_vertex_indices,
            mean_plus_blendshapes, lambda_p, num_shape_coefficients_to_fit)

        # Estimate the blendshape coefficients with the current PCA model estimate:
        current_pca_shape = morphable_model.shape_model.draw_sample(pca_shape_coefficients)
        blendshape_coefficients = blendshape_fitting.fit_blendshapes_to_landmarks_nnls(
            blendshapes, current_pca_shape, front_affine_from_ortho, front_image_points, front_vertex_indices)

        current_combined_shape = current_pca_shape + blendshapes_as_basis.dot(blendshape_coefficients)
        current_mesh = MorphableModel.sample_to_mesh(
            current_combined_shape, morphable_model.color_model.mean, morphable_model.shape_model.triangle_list,
            morphable_model.color_model.triangle_list, morphable_model.texture_coordinates)

        print(np.linalg.norm(pca_shape_coefficients - pre_shape_coefficients))
        if np.linalg.norm(pca_shape_coefficients - pre_shape_coefficients) < 1e-3:
            break

        print('loop {} is finished at time: {}'.format(i + 1, time.time() - start_time))

    current_profile_pose = orthographic_camera_estimation_linear.estimate_orthographic_projection_linear(
        copy.deepcopy(profile_image_points), copy.deepcopy(profile_model_points), True, image_height)
    profile_rendering_params = RenderingParameters.RenderingParameters(
        current_profile_pose, image_width, image_height)

    return current_mesh, front_rendering_params, profile_rendering_params
