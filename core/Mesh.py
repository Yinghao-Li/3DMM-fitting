# -*- coding: utf-8 -*-
"""
@author: Yinghao Li
"""

        
class Mesh:
    """
    This class represents a 3D mesh consisting of vertices, vertex colour
    information and texture coordinates.
    
    Additionally it stores the indices that specify which vertices
    to use to generate the triangle mesh out of the vertices.
    
    Attributes:
        vertices: 3D vertex positions.
        colors: Colour information for each vertex. Expected to be in RGB order.
        texcoords: Texture coordinates for each vertex.
        tvi: Triangle vertex indices
        tci: Triangle color indices
    """
    def __init__(self):
        """
        Constructor for Mesh.
        """
        self.vertices = []
        self.colors = []
        self.texcoords = []
        self.tvi = []
        self.tci = []


def write_obj(mesh, filename):
    """
    Writes the given Mesh to an obj file that for example can be read by MeshLab.
    
    If the mesh contains vertex colour information, it will be written to the obj as well.
    
    Args:
        mesh: The mesh to save as obj.
        filename: Output filename (including ".obj").
    
    Returns:
        None
    """
    assert len(mesh.vertices) == len(mesh.colors) or not mesh.colors
    obj_file = open(filename, 'w')
    
    if not mesh.colors:
        for i in range(len(mesh.vertices)):
            obj_file.write('v ' + str(mesh.vertices[i][0]) + ' ' + str(mesh.vertices[i][1]) + ' ' + 
                           str(mesh.vertices[i][2]) + '\n')
    else:
        for i in range(len(mesh.vertices)):
            obj_file.write('v ' + str(mesh.vertices[i][0]) + ' ' + str(mesh.vertices[i][1]) + ' ' + 
                           str(mesh.vertices[i][2]) + ' ' + str(mesh.colors[i][0]) + ' ' + str(mesh.colors[i][1]) + 
                           str(mesh.colors[i][2]) + '\n')
    
    if mesh.texcoords:
        for tc in mesh.texcoords:
            obj_file.write('vt ' + str(tc[0]) + ' ' + str(tc[1]) + '\n')
    
    for v in mesh.tvi:
        obj_file.write('f ' + str(v[0]+1) + ' ' + str(v[1]+1) + ' ' + str(v[2]+1) + '\n')
    
    obj_file.close()
    return
