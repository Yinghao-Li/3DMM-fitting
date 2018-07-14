# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 09:33:15 2018

@author: For_Gondor
"""


class Landmark:
    """
    Representation of a landmark, consisting of a landmark name and
    coordinates of the given type.
    
    Attributes:
        name: Name of the landmark, often used as identifier.
        coordinates: The position or coordinates of the landmark.
    """
    def __init__(self, name='', coordinates=list()):
        """
        Initialize Landmark
        """
        self.name = name
        self.coordinates = coordinates


# class Profile_landmark:
#     """
#     Representation of a profile landmark, consisting of a name and
#     coordinates of the given type.
#
#     Attributes:
#         name: Name of the landmark, often used as identifier.
#         coordinates: The position or coordinates of the landmark.
#     """
#     def __init__(self, name = '', coordinates = []):
#         """
#         Initialize Landmark
#         """
#         self.name = name
#         self.coordinates = coordinates


def landmark_filter(landmarks, lm_filter):
    """
    Filters the given LandmarkCollection and returns a new LandmarkCollection
    containing all landmarks whose name matches the one given by lm_filter.
    
    !! This function may not work as intention, should be noticed.
    !! Depends on the type of landmarks param.
    
    Args:
        landmarks: The input LandmarkCollection to be filtered.
        lm_filter: A list of landmark names (identifiers) by which the given LandmarkCollection is filtered.
        
    Returns:
        A new, filtered LandmarkCollection.
    """
    # TODO: This function may not work as intention, should be noticed
    filtered_landmarks = []
    for mark in landmarks:
        if mark.name in lm_filter:
            filtered_landmarks.append(mark)
    return filtered_landmarks
