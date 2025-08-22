# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:49:50 2025

@author: hirar
"""

import numpy as np
import ezdxf as ez
from scipy import interpolate as intp


class Line:
    def __init__(self, x, y, lineType):
        self.x = np.array(x) 
        self.y = np.array(y)
        self.lineType = lineType
        self.st = np.array([x[0], y[0]])
        self.ed = np.array([x[-1], y[-1]])
        #self.length = getLength(self.x, self.y, self.lineType)
        #self.func = getInterpolateFunc(self.x, self.y, self.lineType)

class CLine(Line):
    def __init__(self, x, y, lineType):
        super().__init__(x, y, lineType)

class ALine(CLine):
    def __init__(self, x, y, lineType):
        super().__init__(x, y, lineType)    

def getM1(x, y):
    i = 0
    m1 = []
    while i < len(x)-1:
        if not (x[i+1] == x[i]):
            m1.append( (y[i+1]-y[i]) / (x[i+1]-x[i]))
        else:
            m1.append(np.inf)
        i += 1
    
    if not (x[-1] == x[-2]):
        m1.append( (y[-1]-y[-2]) / (x[-1]-x[-2]) )
    else:
        m1.append(np.inf)
    
    return np.array(m1)

def getSita1(x, y):
    i = 0
    sita1 = []
    while i < len(x)-1:
        temp_sita1 = np.arctan2( y[i+1]-y[i], x[i+1]-x[i])
        sita1.append(temp_sita1)
        i += 1
        
    temp_sita1 = np.arctan2( y[-1]-y[-2], x[-1]-x[-2])
    sita1.append(temp_sita1)
    
    return np.array(sita1)

def getM2(x, y):
    i = 0
    m2 = []
    while i < len(x)-1:
        if not (y[i+1] == y[i]):
            m2.append( -(x[i+1]-x[i])/(y[i+1]-y[i]))
        else:
            m2.append(np.inf)
        i += 1
    
    if not (y[-1] == y[-2]):
        m2.append( -(x[-1]-x[-2])/(y[-1]-y[-2]) )
    else:
        m2.append(np.inf)
    
    return np.array(m2)

def getSita2(x, y):
    sita1 = getSita1(x, y)
    sita2 = sita1 + np.pi/2
    return sita2


def removeSamePoint(x, y):
    i = 0
    new_x = []
    new_y = []
    while i < len(x)-1:
        if not ((x[i+1] == x[i]) and (y[i+1] == y[i])):
            new_x.append(x[i])
            new_y.append(y[i])
        i += 1
    
    if not ((x[-1] == x[-2]) and (y[-1] == y[-2])):
        new_x.append(x[-1])
        new_y.append(y[-1])
    
    return np.array(new_x), np.array(new_y)


def getInterpFunc(tck, der):
    def interpFunc(u):
        return intp.splev(u, tck, der)
    return interpFunc


def getSplineData(x, y, dim):
    temp_x, temp_y = removeSamePoint(x, y)
    
    return intp.splprep([temp_x, temp_y],k=dim,s=0)

def importFromText(fileName, lineType):
    text_file = np.genfromtxt(fileName, delimiter = ",", skip_header = 1, dtype = float)
    temp_x = text_file[:,0]
    temp_y = text_file[:,1]
    x, y = removeSamePoint(temp_x, temp_y)
    imported_line = Line(x, y, lineType)
    return imported_line