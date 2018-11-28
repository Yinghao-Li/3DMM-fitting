# -*- coding: utf-8 -*-
"""
@author: Yinghao Li
"""

import numpy as np
import math


class quat:
    """
    quaternion
    """
    def __init__(self, w=0.0, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w


def quat_cast(m):
    r"""
    from glm\gtc\quaternion.cpp
    
    Args:
        m: a 3x3 ratation array
    
    Returns:
        a quat object
    """
    four_x_squared_minus1 = m[0, 0] - m[1, 1] - m[2, 2]
    four_y_squared_minus1 = m[1, 1] - m[0, 0] - m[2, 2]
    four_z_squared_minus1 = m[2, 2] - m[0, 0] - m[1, 1]
    four_w_squared_minus1 = m[0, 0] + m[1, 1] + m[2, 2]
    
    q_list = [four_w_squared_minus1, four_x_squared_minus1, four_y_squared_minus1, four_z_squared_minus1]
    four_biggest_squared_minus1 = max(q_list)
    biggest_index = q_list.index(four_biggest_squared_minus1)
    
    biggest_val = np.sqrt(four_biggest_squared_minus1 + 1) * 0.5
    mult = 0.25 / biggest_val
    
    if biggest_index == 0:
        return quat(biggest_val, (m[2, 1] - m[1, 2]) * mult, (m[0, 2] - m[2, 0]) * mult, (m[1, 0] - m[0, 1]) * mult)
    elif biggest_index == 1:
        return quat((m[2, 1] - m[1, 2]) * mult, biggest_val, (m[1, 0] + m[0, 1]) * mult, (m[0, 2] + m[2, 0]) * mult)
    elif biggest_index == 2:
        return quat((m[0, 2] - m[2, 0]) * mult, (m[1, 0] + m[0, 1]) * mult, biggest_val, (m[2, 1] + m[1, 2]) * mult)
    elif biggest_index == 3:
        return quat((m[1, 0] - m[0, 1]) * mult, (m[0, 2] + m[2, 0]) * mult, (m[2, 1] + m[1, 2]) * mult, biggest_val)
    else:
        assert False


def mat3_cast(q):
    """
    Cast quaternion to 3x3 ndarray
    
    Args:
        q: quat object
        
    Returns:
        a 3x3 ndarray
    """
    qxx = q.x * q.x
    qyy = q.y * q.y
    qzz = q.z * q.z
    qxz = q.x * q.z
    qxy = q.x * q.y
    qyz = q.y * q.z
    qwx = q.w * q.x
    qwy = q.w * q.y
    qwz = q.w * q.z

    result = np.empty([3, 3])
    
    result[0, 0] = 1 - 2 * (qyy + qzz)
    result[1, 0] = 2 * (qxy + qwz)
    result[2, 0] = 2 * (qxz - qwy)
    
    result[0, 1] = 2 * (qxy - qwz)
    result[1, 1] = 1 - 2 * (qxx + qzz)
    result[2, 1] = 2 * (qyz + qwx)
    
    result[0, 2] = 2 * (qxz + qwy)
    result[1, 2] = 2 * (qyz - qwx)
    result[2, 2] = 1 - 2 * (qxx + qyy)
    
    return result


def mat4_cast(q):
    """
    Cast quaternion to 4x4 ndarray
    
    Args:
        q: quat object
        
    Returns:
        a 4x4 ndarray
    """
    result = np.eye(4)
    result[:3, :3] = mat3_cast(q)
    return result


def ortho(left, right, bottom, top):
    """
    Creates a 4x4 matrix for projecting two-dimensional coordinates onto the screen.
    
    Returns:
        a 4x4 ndarray
    """
    result = np.eye(4)
    result[0, 0] = 2 / (right - left)
    result[1, 1] = 2 / (top - bottom)
    result[0, 3] = - (right + left) / (right - left)
    result[1, 3] = - (top + bottom) / (top - bottom)
    return result


def project(obj: np.ndarray, model: np.ndarray, proj: np.ndarray, viewport: np.ndarray) -> np.ndarray:
    """
    Map the specified object coordinates (obj.x, obj.y, obj.z) into window coordinates.
    The near and far clip planes correspond to z normalized device coordinates of -1 and +1 respectively.
    (OpenGL clip volume definition)
    
    From https://github.com/g-truc/glm/blob/master/glm/gtc/matrix_transform.inl
    The C++ programme uses glm to compute the object's projection on screen
    but I did not find any similar python modules, so I have to write it 
    by my own. Considering this, the accuracy could not be guaranteed.
    
    Args:
        obj: length-3 vector.
        model: 4-by-4 matrix.
        proj: 4-by-4 matrix.
        viewport: length-4 vector.
        
    Returns:
        length-3 vector
    """
    tmp = np.hstack([obj, 1.0])
    tmp = model.dot(tmp)
    tmp = proj.dot(tmp)
    
    tmp /= tmp[3]
    tmp = tmp * 0.5 + 0.5
    tmp[0] = tmp[0] * viewport[2] + viewport[0]
    tmp[1] = tmp[1] * viewport[3] + viewport[1]
    
    return tmp[:3]


def project_advance(obj, model, proj, viewport):
    tmp = np.hstack([obj, np.ones([len(obj), 1])])
    tmp = model.dot(tmp.T).T
    tmp = proj.dot(tmp.T).T
    
    tmp /= tmp[:, None, 3]
    tmp = tmp * 0.5 + 0.5
    tmp[:, 0] = tmp[:, 0] * viewport[2] + viewport[0]
    tmp[:, 1] = tmp[:, 1] * viewport[3] + viewport[1]
    
    return tmp[:, :3]


def roll(q):
    """
    Returns roll value of euler angles expressed in radians.
    
    Args:
        q: A quat-type object
            
    Returns:
        Roll value of euler angles expressed in radians.
    """
    return math.atan2(2 * (q.x * q.y + q.w * q.z), q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z)


def pitch(q):
    """
    Returns pitch value of euler angles expressed in radians.
    
    Args:
        q: A quat-type object
    
    Returns:
        Pitch value of euler angles expressed in radians.
    """
    y = 2 * (q.y * q.z + q.w * q.x)
    x = q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z
    
    if y == 0 and x == 0:
        return 2 * math.atan2(q.x, q.w)
    
    return math.atan2(y, x)


def yaw(q):
    """
    Returns yaw value of euler angles expressed in radians.
    
    Args:
        q: A quat-type object
    
    Returns:
        Yaw value of euler angles expressed in radians.
    """
    return math.asin(np.clip(-2 * (q.x * q.z - q.w * q.y), -1, 1))
