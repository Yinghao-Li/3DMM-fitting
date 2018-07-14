# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 18:47:08 2018

@author: For_Gondor
"""

from core import Mesh
import numpy as np
import pickle


def sample_to_mesh(shape_instance, color_instance, tvi, tci, texture_coordinates=None):
    """
    Helper function that creates a Mesh from given shape and colour PCA
    instances. Needs the vertex index lists as well to assemble the mesh -
    and optional texture coordinates.

    If -color_instance is empty, it will create a mesh without vertex colouring.
    Colour values are assumed to be in the range [0, 1] and will be clamped to [0, 1].
    
    Args:
        shape_instance: PCA shape model instance.
        color_instance: PCA colour model instance.
        tvi: Triangle vertex indices.
        tci: Triangle colour indices (usually identical to the vertex indices).
        texture_coordinates: Optional texture coordinates for each vertex.
        
    Return:
        A mesh created from given parameters.
    """
    assert len(shape_instance) == len(color_instance) or len(color_instance) == 0
    
    num_vertices = len(shape_instance) // 3
    mesh = Mesh.Mesh()
    
    # Construct the mesh vertices:
    mesh.vertices = shape_instance.reshape([num_vertices, 3])
    
    # Assign the vertex colour information if it's not a shape-only model:
    if len(color_instance) > 0:
        mesh.colors = color_instance.reshape([num_vertices, 3]).clip(0, 1)
    
    # Assign the triangle lists:
    mesh.tvi = tvi
    mesh.tci = tci
    
    # Texture coordinates, if the model has them:
    if texture_coordinates:
        mesh.texcoords = []
        for i in range(num_vertices):
            mesh.texcoords.append(np.array(texture_coordinates[i]))
            
    return mesh


class MorphableModel:
    """
    Create a Morphable Model from a shape and a colour PCA model, and optional
    texture coordinates.
    
    Attributes:
        shape_model: A PCA model over the shape.
        color_model: A PCA model over the colour (albedo).
        texture_coordinates: Optional texture coordinates for every vertex.
    """
    def __init__(self, shape_model, color_model, texture_coordinates=None):
        """
        Constructor for MorphableModel
        """
        self.shape_model = shape_model
        self.color_model = color_model
        self.texture_coordinates = texture_coordinates
    
    def get_mean(self):
        """
        Returns the mean of the shape- and colour model as a Mesh.
            
        Returns:
            A mesh instance of the mean of the Morphable Model.
        """
        assert len(self.shape_model) == len(self.color_model) or not self.color_model
        
        shape = self.shape_model.mean
        color = self.color_model.mean
        
        mesh = sample_to_mesh(shape, color, self.shape_model.triangle_list,
                              self.color_model.triangle_list, self.texture_coordinates)
        return mesh


def save_model(model, filename):
    """
    Helper method to save a Morphable Model to the harddrive
    
    Args:
        model: The model to be saved.
        filename: Filename for the model.
    """
    file = open(filename, 'wb')
    pickle.dump(model, file, -1)
    file.close()


def load_model(filename: str) -> MorphableModel:
    """
    Helper method to load a Morphable Model from the harddisk.
    
    Args:
        filename: Filename to a model.
        
    Returns:
        The loaded Morphable Model.
    """
    file = open(filename, 'rb')
    model = pickle.load(file)
    file.close()
    return model
