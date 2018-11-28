# -*- coding: utf-8 -*-
"""
@author: Yinghao Li
"""

import json
import toml
import numpy as np
from core import glm


class ModelContour:
    """
    Definition of the vertex indices that define the right and left model contour.
    
    This class holds definitions for the contour (outline) on the right and left
    side of the reference 3D face model. These can be found in the file
    share/model_contours.json. The Surrey model's boundaries are conveniently
    approximately located near the actual 2D image contour, for the front-facing
    contour.
    
    Note: We should extend that to the 1724 model to get a few more points, this
    should improve the contour fitting.
    
    We store r/l separately because we currently only fit to the contour facing the camera.
    Also if we were to fit to the whole contour: Be careful not to just fit to the closest. The
    "invisible" ones behind might be closer on an e.g 90� angle. Store CNT for left/right side separately?
    
    Attributes:
        right_contour: starting from right side, eyebrow-height. (I think the order matters here)
        left_contour: starting from left side, eyebrow-height.
        # 23 = middle, below chin - not included in the contour here
    """
    def __init__(self, right_contour=None, left_contour=None):
        """
        Initialize the ModelContour class
        """
        self.right_contour = right_contour
        self.left_contour = left_contour
    
    def load(self, filename):
        """
        Helper method to load a ModelContour from a json file from the hard drive.
        
        !! Notice: It is different in the ways this function should be called from 
        the original C++ version.
        
        Args:
            filename: Filename to a model.
        
        Returns:
            None
        """
        file = open(filename, 'r')
        outer_dict = json.load(file)
        inner_dict = outer_dict['model_contour']
        
        self.right_contour = inner_dict['right_contour']
        self.left_contour = inner_dict['left_contour']
        
        return None


class ContourLandmarks:
    """
    Defines which 2D landmarks comprise the right and left face contour.
    
    This class holds 2D image contour landmark information. More specifically,
    it defines which 2D landmark IDs correspond to the right contour and which
    to the left. These definitions are loaded from a file, for example from
    the "contour_landmarks" part of share/ibug_to_sfm.txt.
    
    Note: Better names could be ContourDefinition or ImageContourLandmarks, to
    disambiguate 3D and 2D landmarks?
    
    Attributes:
        right_contour: starting from right side, eyebrow-height.
        left_contour: starting from left side, eyebrow-height. Order doesn't matter here.
        # Chin point is not included in the contour here.
    """
    def __init__(self, right_contour=None, left_contour=None):
        """
        Initialize ContourLandmarks
        """
        self.right_contour = right_contour
        self.left_contour = left_contour
        
    def load(self, filename):
        """
        Helper method to load contour landmarks from a text file with landmark
        mappings, like ibug_to_sfm.txt.
        
        Args:
            filename: Filename to a landmark-mapping file.
        
        Returns:
            None
        """
        data = toml.load(filename)
        contour_table = data['contour_landmarks']
        
        # I don't understand why the original programme had to save the values
        # as string type, but as it did, I shall follow it's way here.
        # TODO: Maybe I could change this in the future
        right_contour = contour_table['right']
        self.right_contour = []
        for landmark in right_contour:
            self.right_contour.append(str(landmark))
        
        left_contour = contour_table['left']
        self.left_contour = []
        for landmark in left_contour:
            self.left_contour.append(str(landmark))
            
        return None
    

def select_contour(yaw_angle, contour_landmarks, model_contour):
    """
    Takes a set of 2D and 3D contour landmarks and a yaw angle and returns two
    vectors with either the right or the left 2D and 3D contour indices. This
    function does not establish correspondence between the 2D and 3D landmarks,
    it just selects the front-facing contour. The two returned vectors can thus
    have different size. 
    Correspondence can be established using get_nearest_contour_correspondences().
    
    If the yaw angle is between +-7.5�, both contours will be selected.
    
    Note: Maybe rename to find_nearest_contour_points, to highlight that there is (potentially a lot)
        computational cost involved?
    
    Args:
        yaw_angle:
            Yaw angle in degrees.
        contour_landmarks:
            2D image contour ids of left or right side (for example for ibug landmarks).
        model_contour:
            The model contour indices that should be used/considered to find the closest corresponding 3D vertex.
        
    Returns:
        A pair with two vectors containing the selected 2D image contour landmark ids and the 3D model contour indices.
    """
    model_contour_indices = []
    contour_landmark_identifiers = []
    
    # positive yaw = subject looking to the left
    if yaw_angle >= -7.5:
        # ==> we use the right cnt-lms
        model_contour_indices.extend(model_contour.right_contour)
        contour_landmark_identifiers.extend(contour_landmarks.right_contour)
    if yaw_angle <= 7.5:
        # ==> we use the left cnt-lms
        model_contour_indices.extend(model_contour.left_contour)
        contour_landmark_identifiers.extend(contour_landmarks.left_contour)
    # Note there's an overlap between the angles - if a subject is between +- 7.5, both contours get added.
    return contour_landmark_identifiers, model_contour_indices


def get_nearest_contour_correspondences(landmarks, landmark_contour_identifiers, model_contour_indices,
                                        mesh, view_model, ortho_projection, viewport):
    """
    Given a set of 2D image landmarks, finds the closest (in a L2 sense) 3D vertex
    from a list of vertices pre-defined in \p model_contour. Assumes to be given
    contour correspondences of the front-facing contour.
    
    Note: Maybe rename to find_nearest_contour_points, to highlight that there is (potentially a lot)
        computational cost involved?
    Note: Does ortho_projection have to be specifically orthographic? Otherwise, if it works with perspective too,
    rename to just "projection".
    Note: Actually, only return the vertex id, not the model point as well? Same with get_corresponding_pointset?
    
    Args:
        landmarks:
            All image landmarks.
        landmark_contour_identifiers:
            2D image contour ids of left or right side (for example for ibug landmarks).
        model_contour_indices:
            The model contour indices that should be considered to find the closest corresponding 3D vertex.
        mesh:
            The mesh that's projected to find the nearest contour vertex.
        view_model:
            Model-view matrix of the current fitting to project the 3D model vertices to 2D.
        ortho_projection:
            Projection matrix to project the 3D model vertices to 2D.
        viewport:
            Current viewport to use.
    
    Returns:
        A tuple with the 2D contour landmark points, the corresponding points in the 3D shape model and their
        vertex indices.
    """
    # These are the additional contour-correspondences we're going to find and then use!
    # cnt = contour
    model_points_cnt = []  # the points in the 3D shape model
    vertex_indices_cnt = []  # their vertex indices
    image_points_cnt = []  # the corresponding 2D landmark points
    
    # For each 2D-CNT-LM, find the closest 3DMM-CNT-LM and add to correspondences:
    # Note: If we were to do this for all 3DMM vertices, then ray-casting (i.e. glm::unproject) would be
    # quicker to find the closest vertex
    for ibug_idx in landmark_contour_identifiers:
        # Check if the contour landmark is amongst the landmarks given to us (from detector or ground truth):
        # (Note: Alternatively, we could filter landmarks beforehand and then just loop over landmarks =>
        # means one less function param here. Separate filtering from actual algorithm.)
        # TODO: the whole Landmark class should be replaced by dict in the future
        landmark_dict = {}
        for landmark in landmarks:
            landmark_dict[landmark.name] = landmark.coordinates
        # So it's possible that the function will not return any correspondences.
        if ibug_idx not in landmark_dict.keys():
            continue
        screen_point_2d_contour_landmark = landmark_dict[ibug_idx]
        
        distances_2d = []
        for model_contour_vertex_idx in model_contour_indices:
            vertex = mesh.vertices[model_contour_vertex_idx]
            # TODO: Not sure whether it will work
            proj = glm.project(vertex, view_model, ortho_projection, viewport)
            screen_point_model_contour = proj[:2]
            
            dist = np.linalg.norm(screen_point_model_contour - screen_point_2d_contour_landmark)
            distances_2d.append(dist)
        min_ele = np.min(distances_2d)
        # TODO: Cover the case when cnt_indices_to_use.size() is 0.
        min_ele_idx = distances_2d.index(min_ele)
        the_3dmm_vertex_id_that_is_closest = model_contour_indices[min_ele_idx]
        
        vertex = np.hstack([mesh.vertices[the_3dmm_vertex_id_that_is_closest], 1.0])
        model_points_cnt.append(vertex)
        vertex_indices_cnt.append(the_3dmm_vertex_id_that_is_closest)
        image_points_cnt.append(screen_point_2d_contour_landmark)
        
    return image_points_cnt, model_points_cnt, vertex_indices_cnt


def get_contour_correspondences(landmarks, contour_landmarks, model_contour, yaw_angle, mesh, 
                                view_model, ortho_projection, viewport):
    """
    Given a set of 2D image landmarks, finds the closest (in a L2 sense) 3D vertex
    from a list of vertices pre-defined in \p model_contour. \p landmarks can contain
    all landmarks, and the function will sub-select the relevant contour landmarks with
    the help of the given \p contour_landmarks. This function choses the front-facing
    contour and only fits this contour to the 3D model, since these correspondences
    are approximately static and do not move with changing pose-angle.
    
    It's the main contour fitting function that calls all other functions.
    
    Note: Maybe rename to find_contour_correspondences, to highlight that there is
    (potentially a lot) computational cost involved?
    Note: Does ortho_projection have to be specifically orthographic? Otherwise,
    if it works with perspective too, rename to just "projection".
    
    Args:
        landmarks:
            All image landmarks.
        contour_landmarks:
            2D image contour ids of left or right side (for example for ibug landmarks).
        model_contour:
            The model contour indices that should be considered to find the closest
            corresponding 3D vertex.
        yaw_angle:
            Yaw angle of the current fitting, in degrees. The front-facing contour will
            be chosen depending on this yaw angle.
        mesh:
            The mesh that's used to find the nearest contour points.
        view_model:
            Model-view matrix of the current fitting to project the 3D model vertices to 2D.
        ortho_projection:
            Projection matrix to project the 3D model vertices to 2D.
        viewport:
            Current viewport to use.
    
    Returns:
        A tuple with the 2D contour landmark points, the corresponding points in the 3D shape model
        and their vertex indices.
    """
    # Select which side of the contour we'll use:
    landmark_contour_identifiers, model_contour_indices = select_contour(yaw_angle, contour_landmarks, model_contour)
    
    # For each 2D contour landmark, get the corresponding 3D vertex point and vertex id:
    # TODO: Loop here instead of calling this function where we have no idea what it's doing? What does
    # its documentation say?
    return get_nearest_contour_correspondences(landmarks, landmark_contour_identifiers, model_contour_indices,
                                               mesh, view_model, ortho_projection, viewport)
