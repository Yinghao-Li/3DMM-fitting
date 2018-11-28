# -*- coding: utf-8 -*-
"""
@author: Yinghao Li
"""

import toml  # see https://github.com/uiri/toml


class LandmarkMapper:
    """
    Represents a mapping from one kind of landmarks
    to a different format (e.g. model vertices).
    
    When fitting the 3D model to an image, a correspondence must
    be known from the 2D image landmarks to 3D vertex points in
    the Morphable Model. The 3D model defines all its points in
    the form of vertex ids.
    These mappings are stored in a file, see the share folder for
    an example for mapping 2D ibug landmarks to 3D model vertex indices.
    
    The LandmarkMapper thus has two main use cases:
        - Mapping 2D landmark points to 3D vertices
        - Converting one set of 2D landmarks into another set of 2D
          landmarks with different identifiers.
          
    Attributes:
        landmark_mappings:
            Mapping from one landmark name to a name in a different format.
    """
    def __init__(self, filename):
        """
        Initialize LandmarkMapper
        """
        # extracting the data
        data = toml.load(filename)
        self.landmark_mappings = data['landmark_mappings']
    
    def convert(self, landmark_name):
        """
        Converts the given landmark name to the mapped name.
        !! Seems useless in Python as its function is covered by dict type
        
        Args:
            landmark_name:
                A landmark name to convert.
            
        Returns:
            The mapped landmark name if a mapping exists, an empty optional otherwise.
        """
        if not landmark_name:
            return None
        
        if landmark_name in self.landmark_mappings.keys():
            return self.landmark_mappings[landmark_name]
        else:
            return None
            
    def num_mappings(self):
        """
        Returns the number of loaded landmark mappings.
        
        Returns:
            The number of landmark mappings.
        """
        return len(self.landmark_mappings)


class ProfileLandmarkMapper:
    """
    Represents the mapping from the profile landmarks defined by myself to
    three-dimentional vertices
    """
    def __init__(self, filename):
        all_mapper = toml.load(filename)
        self.left_mapper = all_mapper['left_profile_mappings']
        self.right_mapper = all_mapper['right_profile_mappings']
