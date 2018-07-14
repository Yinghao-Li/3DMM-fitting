# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 11:25:43 2018

@author: For_Gondor
"""

import numpy as np
from core import glm, utils, RenderingParameters, Mesh

from scipy.spatial import KDTree


def ray_triangle_intersect(ray_origin, ray_direction, v0, v1, v2, enable_backculling):
    r"""
    Computes the intersection of the given ray with the given triangle.
    
    Uses the Muler-Trumbore algorithm algorithm "Fast Minimum Storage
    Ray/Triangle Intersection". Independent implementation, inspired by:
    http://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
    The default eps (1e-6f) is from the paper.
    When culling is on, rays intersecting triangles from the back will be discarded -
    otherwise, the triangles normal direction w.r.t. the ray direction is just ignored.
    
    Note: The use of optional might turn out as a performance problem, as this
    function is called loads of time - how costly is it to construct a boost::none optional?
    
    Args:
        ray_origin: Ray origin.
        ray_direction: Ray direction.
        v0: First vertex of a triangle.
        v1: Second vertex of a triangle.
        v2: Third vertex of a triangle.
        enable_backculling: When culling is on, rays intersecting triangles from the back will be discarded.
    
    Returns:
        Whether the ray intersects the triangle, and if yes, including the distance.
    """
    epsilon = 1e-6
    
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    
    pvec = np.cross(ray_direction, v0v2)
    
    det = np.dot(v0v1, pvec)
    if enable_backculling:
        # If det is negative, the triangle is back-facing.
        # If det is close to 0, the ray misses the triangle.
        if det < epsilon:
            return None
    else:
        # If det is close to 0, the ray and triangle are parallel.
        if np.abs(det) < epsilon:
            return None
    inv_det = 1.0 / det
    
    tvec = ray_origin - v0
    u = np.dot(tvec, pvec) * inv_det
    if u < 0 or u > 1:
        return None
    
    qvec = np.cross(tvec, v0v1)
    v = np.dot(ray_direction, qvec) * inv_det
    if v < 0 or u + v > 1:
        return None
    
    t = np.dot(v0v2, qvec) * inv_det
    return t


def ray_triangle_intersect_advance(ray_origin, ray_direction, v0, v1, v2):

    epsilon = 1e-6

    v0v1 = v1 - v0
    v0v2 = v2 - v0

    pvec = np.cross(ray_direction, v0v2)
    det = np.sum(v0v1 * pvec, axis=1)
    inv_det = 1.0 / det

    tvec = ray_origin - v0
    u = np.sum(tvec * pvec, axis=1) * inv_det

    qvec = np.cross(tvec, v0v1)
    v = np.sum(ray_direction * qvec, axis=1) * inv_det

    t = np.sum(v0v2 * qvec, axis=1) * inv_det

    p1 = np.where(np.abs(det) >= epsilon)[0]
    p2_1 = np.where(u >= 0)[0]
    p2_2 = np.where(u <= 1)[0]
    p2 = np.intersect1d(p2_1, p2_2)
    p3_1 = np.where(v >= 0)[0]
    p3_2 = np.where(u + v < 1)[0]
    p3 = np.intersect1d(p3_1, p3_2)
    p4 = np.where(t >= 1e-4)[0]
    inter = np.intersect1d(p1, p2)
    inter = np.intersect1d(inter, p3)
    inter = np.intersect1d(inter, p4)

    return len(inter) == 0


def ray_triangle_intersect_advance1(ray_origins, ray_direction, v0, v1, v2):

    epsilon = 1e-6

    v0v1 = v1 - v0
    v0v2 = v2 - v0

    pvec = np.cross(ray_direction, v0v2)
    det = np.sum(v0v1 * pvec, axis=1)
    inv_det = 1.0 / det

    p1 = np.where(np.abs(det) >= epsilon)[0]
    visibility = []
    print(len(ray_origins))

    for ray_origin in ray_origins:

        tvec = ray_origin - v0
        u = np.sum(tvec * pvec, axis=1) * inv_det

        qvec = np.cross(tvec, v0v1)
        v = np.sum(ray_direction * qvec, axis=1) * inv_det

        t = np.sum(v0v2 * qvec, axis=1) * inv_det

        p2_1 = np.where(u >= 0)[0]
        p2_2 = np.where(u <= 1)[0]
        p2 = np.intersect1d(p2_1, p2_2)
        p3_1 = np.where(v >= 0)[0]
        p3_2 = np.where(u + v < 1)[0]
        p3 = np.intersect1d(p3_1, p3_2)
        p4 = np.where(t >= 1e-4)[0]
        inter = np.intersect1d(p1, p2)
        inter = np.intersect1d(inter, p3)
        inter = np.intersect1d(inter, p4)

        visibility.append(len(inter) == 0)

    return visibility


def ray_triangle_intersect_advance2(ray_origins, ray_direction, v0, v1, v2):

    epsilon = 1e-6

    v0v1 = v1 - v0
    v0v2 = v2 - v0

    pvec = np.cross(ray_direction, v0v2)
    det = np.sum(v0v1 * pvec, axis=1)
    inv_det = 1.0 / det

    p1 = np.where(np.abs(det) >= epsilon, True, False)
    visibility = []
    print(len(ray_origins))

    for ray_origin in ray_origins:

        tvec = ray_origin - v0
        u = np.sum(tvec * pvec, axis=1) * inv_det

        qvec = np.cross(tvec, v0v1)
        v = np.sum(ray_direction * qvec, axis=1) * inv_det

        t = np.sum(v0v2 * qvec, axis=1) * inv_det

        p2_1 = np.where(u >= 0, True, False)
        p2_2 = np.where(u <= 1, True, False)
        p3_1 = np.where(v >= 0, True, False)
        p3_2 = np.where(u + v < 1, True, False)
        p4 = np.where(t >= 1e-4, True, False)
        p = p1 * p2_1 * p2_2 * p3_1 * p3_2 * p4
        visibility.append(not p.any())

    return visibility


def occluding_boundary_vertices(mesh, edge_topology, r):
    """
    Computes the vertices that lie on occluding boundaries, given a particular pose.
    
    This algorithm computes the edges that lie on occluding boundaries of the mesh.
    It performs a visibility text of each vertex, and returns a list of the (unique)
    vertices that make the boundary edges.
    An edge is defined as the line whose two adjacent faces normals flip the sign.
    
    Args:
        mesh:
            The mesh to use.
        edge_topology:
            The edge topology of the given mesh.
        r:
            The rotation (pose) under which the occluding boundaries should be computed.
        
    Returns:
        A vector with unique vertex id's making up the edges.
    """
    # Rotate the mesh:
    rotated_vertices = r.dot(np.hstack([mesh.vertices, np.ones([len(mesh.vertices), 1])]).T).T[:, :3]
    v0 = np.asarray([rotated_vertices[tri[0]] for tri in mesh.tvi])
    v1 = np.asarray([rotated_vertices[tri[1]] for tri in mesh.tvi])
    v2 = np.asarray([rotated_vertices[tri[2]] for tri in mesh.tvi])
    
    # Compute the face normals of the rotated mesh
    facenormals = utils.compute_face_normal(v0, v1, v2)
    # Find occluding edges:
    occluding_edges_indices = []
    for edge_idx in range(len(edge_topology.adjacent_faces)):
        # For each edge... Ef contains the indices of the two adjacent faces
        edge = edge_topology.adjacent_faces[edge_idx]
        # Edges with a zero index lie on the mesh boundary, i.e. they are only
        # adjacent to one face.
        # TODO: Need to change this if we use 0-based indexing!        
        if edge[0] == 0:
            continue
        # Compute the occluding edges as those where the two adjacent face normals
        # differ in the sign of their z-component:
        # Changing from 1-based indexing to 0-based!
        if np.sign(facenormals[edge[0] - 1, 2]) != np.sign(facenormals[edge[1] - 1, 2]):
            # It's an occluding edge, store the index:
            occluding_edges_indices.append(edge_idx)

    # Select the vertices lying at the two ends of the occluding edges and remove duplicates:
    # (This is what EdgeTopology.adjacent_vertices is needed for).
    occluding_vertices = []  # The model's contour vertices
    for edge_idx in occluding_edges_indices:
        # Changing from 1-based indexing to 0-based!
        occluding_vertices.append(edge_topology.adjacent_vertices[edge_idx][0] - 1)
        occluding_vertices.append(edge_topology.adjacent_vertices[edge_idx][1] - 1)
    # Remove duplicate vertex id's (std::unique works only on sorted sequences):
    occluding_vertices = list(set(occluding_vertices))
    
    # Perform ray-casting to find out which vertices are not visible (i.e. self-occluded):
    ray_direction = np.array([0.0, 0.0, 1.0])  # we shoot the ray from the vertex towards the camera
    # visibility = []
    # for vertex_idx in occluding_vertices:
    #     ray_origin = rotated_vertices[vertex_idx]
    #     visibility.append(ray_triangle_intersect_advance(ray_origin, ray_direction, v0, v1, v2, False))
    # visibility = [ray_triangle_intersect_advance(rotated_vertices[vertex_idx], ray_direction, v0, v1, v2)
    #               for vertex_idx in occluding_vertices]
    ray_origins = np.asarray([rotated_vertices[vertex_idx] for vertex_idx in occluding_vertices])
    visibility = ray_triangle_intersect_advance2(ray_origins, ray_direction, v0, v1, v2)

    # Remove vertices from occluding boundary list that are not visible:
    final_vertex_ids = [occluding_vertices[i] for i in range(len(occluding_vertices)) if visibility[i]]
            
    return final_vertex_ids


def find_occluding_edge_correspondences(mesh, edge_topology, rendering_parameters, image_edges,
                                        distance_threshold=64.0):
    """
    For a given list of 2D edge points, find corresponding 3D vertex IDs.
    
    This algorithm first computes the 3D mesh's occluding boundary vertices under
    the given pose. Then, for each 2D image edge point given, it searches for the
    closest 3D edge vertex (projected to 2D). Correspondences lying further away
    than \c distance_threshold (times a scale-factor) are discarded.
    It returns a list of the remaining image edge points and their corresponding
    3D vertex ID.
    
    The given \c rendering_parameters camery_type must be CameraType::Orthographic.
    
    The units of \c distance_threshold are somewhat complicated. The function
    uses squared distances, and the \c distance_threshold is further multiplied
    with a face-size and image resolution dependent scale factor.
    It's reasonable to use correspondences that are 10 to 15 pixels away on a
    1280x720 image with s=0.93. This would be a distance_threshold of around 200.
    64 might be a conservative default.
    
    Args:
        mesh:
            The 3D mesh.
        edge_topology:
            The mesh's edge topology (used for fast computation).
        rendering_parameters:
            Rendering (pose) parameters of the mesh.
        image_edges:
            A list of points that are edges.
        distance_threshold:
            All correspondences below this threshold.
        
    Returns:
        A pair consisting of the used image edge points and their associated 3D vertex index.
    """
    assert rendering_parameters.get_camera_type() == RenderingParameters.CameraType['Orthographic']

    # If there are no image_edges given, there's no point in computing anything:
    if len(image_edges) == 0:
        return None, None
    
    # Compute vertices that lye on occluding boundaries:
    occluding_vertices = occluding_boundary_vertices(
        mesh, edge_topology, glm.mat4_cast(rendering_parameters.get_rotation()))
    occluding_vertices.sort()
    
    # Project these occluding boundary vertices from 3D to 2D:
    occ_ver = np.array([mesh.vertices[v] for v in occluding_vertices])
    model_edges_projected = glm.project_advance(
        occ_ver, rendering_parameters.get_modelview(), rendering_parameters.get_projection(),
        RenderingParameters.get_opencv_viewport(rendering_parameters.get_screen_width(),
                                                rendering_parameters.get_screen_height()))[:, :2]
    # Find edge correspondences:

    # Build a kd-tree and use nearest neighbour search:
    tree = KDTree(np.array(image_edges))
    
    # will contain [distance (L2) , index to the 2D edge in 'image_edges']
    idx_d = np.array(tree.query(np.array(model_edges_projected))).T
    # Filter edge matches:
    # We filter below by discarding all correspondence that are a certain distance apart.
    # We could also (or in addition to) discard the worst 5% of the distances or something like that.
    
    # Filter and store the image (edge) points with their corresponding vertex id:
    vertex_indices = []
    image_points = []
    assert len(occluding_vertices) == len(idx_d)
    for i in range(len(occluding_vertices)):
        # This might be a bit of a hack - we recover the "real" scaling from the SOP estimate
        ortho_scale = rendering_parameters.get_screen_width() / rendering_parameters.get_frustum().r
        # I think multiplying by the scale is good here and gives us invariance w.r.t.
        # the image resolution and face size.
        if idx_d[i, 0] ** 2 <= distance_threshold * ortho_scale:
            edge_point = image_edges[int(idx_d[i, 1])]
            # Store the found 2D edge point, and the associated vertex id:
            vertex_indices.append(occluding_vertices[i])
            image_points.append(edge_point)
    
    return image_points, vertex_indices


def visible(mesh: Mesh.Mesh, r: np.ndarray) ->np.ndarray:
    rotated_vertices = r.dot(np.hstack([mesh.vertices, np.ones([len(mesh.vertices), 1])]).T).T[:, :3]
    z = rotated_vertices[:, 2]
    uv = rotated_vertices[:, :]
    uv[:, 0] = uv - np.min(uv[:, 0], axis=0)
    uv = uv + 1
    uv = uv / np.max(uv) * 1000
    width = 1000
    height = 1000
    faces = np.array(mesh.tvi)
    v1 = faces[:, 0]
    v2 = faces[:, 1]
    v3 = faces[:, 2]
    nfaces = np.shape(faces)[0]
    x = np.c_[uv[v1, 0], uv[v2, 0], uv[v3, 0]]
    y = np.c_[uv[v1, 1], uv[v2, 1], uv[v3, 1]]
    minx = np.ceil(x.min(1))
    maxx = np.floor(x.max(1))
    miny = np.ceil(y.min(1))
    maxy = np.floor(y.max(1))

    del x, y

    minx = np.clip(minx, 0, width - 1)
    maxx = np.clip(maxx, 0, width - 1)
    miny = np.clip(miny, 0, height - 1)
    maxy = np.clip(maxy, 0, height - 1)

    [rows, cols] = np.meshgrid(np.linspace(1, 1000, num=1000), np.linspace(1, 1000, num=1000))
    zbuffer = -np.inf(height, width)
    fbuffer = np.zeros(height, width)

    for i in range(nfaces):
        if minx[i] <= maxx[i] and miny[i] <= maxy[i]:
            px = rows[miny[i]: maxy[i], minx[i]: maxx[i]]
            py = cols[miny[i]: maxy[i], minx[i]: maxx[i]]
            px = px[:]
            py = py[:]

            e0 = uv[v1[i], :]
            e1 = uv[v2[i], :]
            e2 = uv[v3[i], :]

            det = e1[0] * e2[1] - e1[1] * e2[0]
            tmpx = px - e0[0]
            tmpy = py - e0[1]
            a = (tmpx * e2[1] - tmpy * e2[0]) / det
            b = (tmpx * e1[0] - tmpy * e1[1]) / det

            test = a >= 0 & b >= 0 & a + b <= 1

            if np.any(test):
                px = px[test]
                py = py[test]

                w2 = a[test]
                w3 = b[test]
                w1 = 1 - w3 - w2
                pz = z[v1[i]] * w1 + z[v2[i]] * w2 + z[v3[i]] * w3

                if pz > zbuffer[py, px]:
                    zbuffer[py, px] = pz
                    fbuffer[py, px] = i
    test = fbuffer != 0
    f = np.unique(fbuffer[test])
    v = np.unique(np.r_(v1[f], v2[f], v3[f]))
    return v
