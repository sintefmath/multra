from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import numpy as np
from copy import copy as cp

texture = 0
width = 640
height = 480

def initGL():

    glutInit()
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)

    global width
    global height
    glutInitWindowSize(width,height)
    if width>height:
        glViewport(int((width-height)/2),0,height,height)
    else:
        glViewport(0,int((height-width)/2),width,width)
    glutCreateWindow("Traffic Simulation")

    glClear( GL_COLOR_BUFFER_BIT );
    glClearColor(1.0,1.0,1.0,1.0)
    glColor3f(0.0,0.0, 0.0)
    global texture
    texture = glGenTextures(1);
    glBindTexture(GL_TEXTURE_1D, texture);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_1D, 0);

    glClear( GL_COLOR_BUFFER_BIT );

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho(-2,2,-2,2,-2,2)

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

def draw(roads, t, o, Tinterval):
    global width
    global height
    n_height = glutGet(GLUT_WINDOW_HEIGHT);
    n_width = glutGet(GLUT_WINDOW_WIDTH);
    if (n_height != height or n_width!= width):
        width = n_width
        height = n_height
        if width>height:
            glViewport(int((width-height)/2),0,height,height)
        else:
            glViewport(0,int((height-width)/2),width,width)
        glClear(GL_COLOR_BUFFER_BIT)
    #glColor3f(0.0,0.0, 0.0)
    glColor3ub( 255, 255, 255 );
    glEnable( GL_TEXTURE_1D );
    glBindTexture( GL_TEXTURE_1D, texture );
    #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    #glEnable(GL_BLEND);
    for i in range(len(roads)):
        r=cp(roads[i].rho[o:-o][0])
        tex=np.zeros((roads[i].n, 1))
        red=0
        green=1
        blue=2
        yellow = np.zeros(roads[i].n)
        #tex[:,green] = -2*r+1
        #tex[:,red] = 2*r-1
        tex[:,red] = r
        #tex[:,green] = r
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RED, roads[i].n, 0, GL_RED, GL_FLOAT, tex);

        glLineWidth( 10 );
        glEnable(GL_LINE_SMOOTH);
        glBegin( GL_LINES );
        glTexCoord1i( 0 );
        glVertex2f( roads[i].geo_IN[0], roads[i].geo_IN[1]);
        glTexCoord1i( 1 );
        glVertex2f( roads[i].geo_OUT[0], roads[i].geo_OUT[1]);
        glEnd();

    glFlush()

