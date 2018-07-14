# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:33:50 2018

@author: For_Gondor
"""

import json


class EdgeTopology:
    """
     A struct containing a 3D shape model's edge topology.
     
     This struct contains all edges of a 3D mesh, and for each edge, it
     contains the two faces and the two vertices that are adjacent to that
     edge. This is used in the iterated closest edge fitting (ICEF).
     
     Note: The indices are 1-based, so 1 needs to be subtracted before using
     them as mesh indices. An index of 0 as first array element means that
     it's an edge that lies on the mesh boundary, i.e. they are only
     adjacent to one face.
     
     We should explore a less error-prone way to store this data, but that's
     how it is done in Matlab by the original code.
     
     adjacent_faces.size() is equal to adjacent_vertices.size().
    """
    def __init__(self):
        """
        Initialize EdgeTopology
        """
        self.adjacent_faces = []
        self.adjacent_vertices = []


def save_edge_topology(edge_topology, filename):
    """
    Saves a 3DMM edge topology file to a json file.
    
    Args:
        edge_topology: A model's edge topology.
        filename: The file to write.
    
    Returns:
        None
    """
    outer_dict = {}
    inner_dict = {'adjacent_faces': edge_topology.adjacent_faces, 'adjacent_vertices': edge_topology.adjacent_vertices}
    outer_dict['edge_topology'] = inner_dict
    
    file = open(filename, 'w')
    json.dump(outer_dict, file, indent=4)
    file.close()


def load_edge_topology(filename):
    """
    Load a 3DMM edge topology file from a json file.
    
    Args:
        filename: The file to load the edge topology from.
        
    Returns:
        An EdgeTopology object containing the edge topology.
    """
    file = open(filename, 'r')
    outer_dict = json.load(file)
    inner_dict = outer_dict['edge_topology']
    
    edge_topology = EdgeTopology()
    edge_topology.adjacent_faces = inner_dict['adjacent_faces']
    edge_topology.adjacent_vertices = inner_dict['adjacent_vertices']
    return edge_topology
