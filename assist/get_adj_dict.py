# -*- coding: utf-8 -*-
"""
@author: Yinghao Li
"""

import sys
import numpy as np
import pickle

sys.path.append('..')
from core import MorphableModel, EdgeTopology

py_model = MorphableModel.load_model(r"..\py_share\py_sfm_shape_3448.bin")
vertices = py_model.shape_model.mean.reshape([-1, 3])

triangles = py_model.shape_model.triangle_list

py_edge_topology = EdgeTopology.load_edge_topology(r'..\py_share\py_sfm_3448_edge_topology.json')
adj_ver = np.array(py_edge_topology.adjacent_vertices)
adj_ver -= 1
adj_dict = {}
for i in range(len(vertices)):
    adj_dict[i] = []

for vers in adj_ver:
    adj_dict[vers[0]].append(vers[1])
    adj_dict[vers[1]].append(vers[0])

file = open(r'..\py_share\adj_dict_3448.pkl', 'wb')
pickle.dump(adj_dict, file, -1)
file.close()
