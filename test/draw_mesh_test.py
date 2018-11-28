# -*- coding: utf-8 -*-
"""
@author: Yinghao Li
"""

from core import MorphableModel
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys
import numpy as np
import pickle

ESCAPE = b'\x1b'

# Number of the glut window.
window = 0
yaw = 0.0
pitch = 0.0
x = 0.0
z = -100.0
point = 114
adj_idx = 0
draw_mode = GL_LINES

pkl_file = open(r'..\py_share\adj_dict_3448.pkl', 'rb')
adj_dict = pickle.load(pkl_file)
pkl_file.close()

py_model = MorphableModel.load_model(r"..\py_share\py_sfm_shape_3448.bin")
vertices = py_model.shape_model.mean.reshape([-1, 3])

triangles = py_model.shape_model.triangle_list

depth_max = np.max(vertices[:, 2])
depth_min = np.min(vertices[:, 2])
colors = (vertices[:, 2] - depth_min) * 1 / (depth_max - depth_min)
colors = np.vstack([np.zeros(len(colors)), colors, 1 - colors]).T


# A general OpenGL initialization function.  Sets all of the initial parameters.
def init_gl(width, height):  # We call this right after our OpenGL window is created.
    glClearColor(0.0, 0.0, 0.0, 0.0)  # This Will Clear The Background Color To Black
    glClearDepth(1.0)  # Enables Clearing Of The Depth Buffer
    glDepthFunc(GL_LESS)  # The Type Of Depth Test To Do
    glEnable(GL_DEPTH_TEST)  # Enables Depth Testing
    glShadeModel(GL_SMOOTH)  # Enables Smooth Color Shading

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()  # Reset The Projection Matrix
    # Calculate The Aspect Ratio Of The Window
    gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)

    glMatrixMode(GL_MODELVIEW)


# The function called when our window is resized (which shouldn't happen if you enable fullscreen, below)
def resize_scene(width, height):
    if height == 0:						# Prevent A Divide By Zero If The Window Is Too Small
        height = 1

    glViewport(0, 0, width, height)		# Reset The Current Viewport And Perspective Transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width) / float(height), 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)


# The main drawing function.
def draw_gl_scene():
    # global rtri, rquad
    global yaw, pitch, x, z, draw_mode, adj_idx

    # Clear The Screen And The Depth Buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glLoadIdentity()					# Reset The View

    # Move Left 1.5 units and into the screen 6.0 units.
    glTranslatef(x, 0.0, z)

    # We have smooth color mode on, this will blend across the vertices.
    # Draw a triangle rotated on the Y axis.
    glRotatef(yaw, 0.0, 1.0, 0.0)      # Rotate
    glRotatef(pitch, 1.0, 0.0, 0.0)
    for triangle in triangles:
        glBegin(draw_mode)                 # Start drawing a polygon
        for ver in triangle:
            if ver == point:
                glColor3f(1.0, 1.0, 1.0)
                glVertex3f(vertices[ver, 0], vertices[ver, 1], vertices[ver, 2])
            elif ver == adj_dict[point][adj_idx]:
                glColor3f(1.0, 0.0, 0.0)
                glVertex3f(vertices[ver, 0], vertices[ver, 1], vertices[ver, 2])
            else:
                glColor3f(colors[ver, 0], colors[ver, 1], colors[ver, 2])
                glVertex3f(vertices[ver, 0], vertices[ver, 1], vertices[ver, 2])
                # print(vertices[ver])
        glEnd()                             # We are done with the polygon

    #  since this is double buffered, swap the buffers to display what just got drawn.
    glutSwapBuffers()


# The function called whenever a key is pressed. Note the use of Python tuples to pass in: (key, x, y)
def key_pressed(*args):
    global yaw, pitch, x, z, draw_mode, point, adj_idx
    # If escape is pressed, kill everything.
    # print(args)
    if args[0] is ESCAPE:
        sys.exit()
    elif args[0] is b'w':
        pitch += 10
    elif args[0] is b's':
        pitch -= 10
    elif args[0] is b'a':
        yaw += 10
    elif args[0] is b'd':
        yaw -= 10
    elif args[0] is b'i':
        z += 10
    elif args[0] is b'k':
        z -= 10
    elif args[0] is b'j':
        x += 10
    elif args[0] is b'l':
        x -= 10
    elif args[0] is b'm':
        if draw_mode is GL_LINES:
            draw_mode = GL_TRIANGLES
        else:
            draw_mode = GL_LINES
    elif args[0] is b'4':
        adj_idx += 1
        if adj_idx >= len(adj_dict[point]):
            adj_idx = 0
    elif args[0] is b'6':
        adj_idx -= 1
        if adj_idx < 0:
            adj_idx = len(adj_dict[point]) - 1
    elif args[0] is b'5':
        point = adj_dict[point][adj_idx]
        adj_idx = 0
        print(point)
    # print(adj_idx)
    draw_gl_scene()


def main():
    global window
    glutInit(sys.argv)

    # Select type of Display mode:
    #  Double buffer
    #  RGBA color
    # Alpha components supported
    # Depth buffer
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

    # get a 640 x 480 window
    glutInitWindowSize(640, 480)

    # the window starts at the upper left corner of the screen
    glutInitWindowPosition(0, 0)

    # Okay, like the C version we retain the window id to use when closing, but for those of you new
    # to Python (like myself), remember this assignment would make the variable local and not global
    # if it weren't for the global declaration at the start of main.
    window = glutCreateWindow("Morphable Model")

    # Register the drawing function with glut, BUT in Python land, at least using PyOpenGL, we need to
    # set the function pointer and invoke a function to actually register the callback, otherwise it
    # would be very much like the C version of the code.
    glutDisplayFunc(draw_gl_scene)

    # Uncomment this line to get full screen.
    # glutFullScreen()

    # When we are doing nothing, redraw the scene.
    # glutIdleFunc(draw_gl_scene)

    # Register the function called when our window is resized.
    glutReshapeFunc(resize_scene)

    # Register the function called when the keyboard is pressed.
    glutKeyboardFunc(key_pressed)

    # Initialize our window.
    init_gl(640, 480)

    # Start Event Processing Engine
    glutMainLoop()


# Print message to console, and kick off the main to get it rolling.
print("Hit ESC key to quit.")
main()
