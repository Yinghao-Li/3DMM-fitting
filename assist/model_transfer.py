# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 18:11:15 2018

@author: For_Gondor
"""

import eos
import os
import pickle
import json

os.chdir('..')

from core import Blendshape, MorphableModel, PcaModel

# 分别转换share/sfm_shape_3448.bin和share/expression_blendshapes_3448.bin文件
model = eos.morphablemodel.load_model("share/sfm_shape_3448.bin")
blendshapes = eos.morphablemodel.load_blendshapes("share/expression_blendshapes_3448.bin")

shape_model = model.get_shape_model()
color_model = model.get_color_model()
texture_coordinates = model.get_texture_coordinates()

shape_mean = shape_model.get_mean()
shape_orthonormal_pca_basis = shape_model.get_orthonormal_pca_basis()
shape_eigenvalues = shape_model.get_eigenvalues()
shape_triangle_list = shape_model.get_triangle_list()
py_shape_model = PcaModel.PcaModel(shape_mean, shape_orthonormal_pca_basis, shape_eigenvalues, shape_triangle_list)

color_mean = color_model.get_mean()
color_orthonormal_pca_basis = color_model.get_orthonormal_pca_basis()
color_eigenvalues = color_model.get_eigenvalues()
color_triangle_list = color_model.get_triangle_list()
py_color_model = PcaModel.PcaModel(color_mean, color_orthonormal_pca_basis, color_eigenvalues, color_triangle_list)

py_texture_coordinates = model.get_texture_coordinates()

py_model = MorphableModel.MorphableModel(py_shape_model, py_color_model, py_texture_coordinates)

MorphableModel.save_model(py_model, 'py_share/py_sfm_shape_3448.bin')

py_blendshapes = []
for blendshape in blendshapes:
    py_blendshape = Blendshape.Blendshape(blendshape.name, blendshape.deformation)
    py_blendshapes.append(py_blendshape)

o = open('py_share/py_expression_blendshapes_3448.bin', 'wb', -1)
pickle.dump(py_blendshapes, o)
o.close()

# 转换share/sfm_3448_edge_topology.json文件
file = open('share/sfm_3448_edge_topology.json', 'r')
outer_dict = json.load(file)
inner_dict = outer_dict['edge_topology']

aj = []
av = []
for dic in inner_dict['adjacent_faces']:
    aj.append([dic['value0'], dic['value1']])
for dic in inner_dict['adjacent_vertices']:
    av.append([dic['value0'], dic['value1']])

inner_dict = {'adjacent_faces': aj, 'adjacent_vertices': av}
outer_dict = {'edge_topology': inner_dict}

file = open('py_share/py_sfm_3448_edge_topology.json', 'w')
json.dump(outer_dict, file, indent=4)
file.close()
