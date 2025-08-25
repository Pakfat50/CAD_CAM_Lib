# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:49:50 2025

@author: hirar
"""

import numpy as np
import ezdxf as ez
from scipy import interpolate as intp
from scipy.optimize import fsolve
from scipy.optimize import fmin
import traceback
from matplotlib import pyplot as plt

# グローバル変数
N_FILLET_INTERPORATE = 10 # フィレット点数

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

def getCrossPointFromPoint(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y):
    # https://imagingsolution.blog.fc2.com/blog-entry-137.html
    s1 = ((p4_x - p2_x) * (p1_y - p2_y) - (p4_y - p2_y) * (p1_x - p2_x)) / 2.0
    s2 = ((p4_x - p2_x) * (p2_y - p3_y) - (p4_y - p2_y) * (p2_x - p3_x)) / 2.0
    
    c1_x = p1_x + (p3_x - p1_x) * (s1 / (s1 + s2))
    c1_y = p1_y + (p3_y - p1_y) * (s1 / (s1 + s2))
    
    return c1_x, c1_y

def getCrossPointFromLines(a, b, c, d):
    # https://mathwords.net/nityokusenkoten
    c1_x = (d-b)/(a-c)
    c1_y = (a*d - b*c)/(a-c)
    return c1_x, c1_y

def getCrossPointFromCurveLine(line_func, curve_func, u0):
    def solver(u):
        p = curve_func(u)
        x_curve = p[0]
        y_curve = p[1]
        y_line = line_func(x_curve)
        y_err = y_curve-y_line
        return y_err
    
    u_root = fsolve(solver, u0)
    p_root = curve_func(u_root)
    x_root = p_root[0]
    y_root = p_root[1]
    
    return u_root, x_root, y_root


def getFiletSitaArray(sita1, sita2):
    if sita2-sita1 <= np.pi and sita2-sita1 > -np.pi:
        sita_array = np.linspace(sita1, sita2, N_FILLET_INTERPORATE)
    elif sita2-sita1 > np.pi:
        sita_array = np.linspace(sita1, sita2-2*np.pi, N_FILLET_INTERPORATE)
    else:
        sita_array = np.linspace(sita1, sita2+2*np.pi, N_FILLET_INTERPORATE)
    return sita_array   

def filetLines(l0_x, l0_y, l1_x, l1_y, r):
    try:
        # l0とl1のなす角sitaを内積により求める
        # https://w3e.kanazawa-it.ac.jp/math/category/vector/henkan-tex.cgi?target=/math/category/vector/naiseki-wo-fukumu-kihonsiki.html&pcview=2
        a1 = l0_x[1] - l0_x[0]
        a2 = l0_y[1] - l0_y[0]
        b1 = l1_x[0] - l1_x[1]
        b2 = l1_y[0] - l1_y[1]
            
        sita = np.arccos( (a1*b1 + a2*b2)/(np.sqrt(a1**2 + a2**2) * np.sqrt(b1**2 + b2**2)) ) / 2.0

        # l0がx軸となす角をarctan2により求める
        alpha = np.arctan2(a2, a1)
        
        # l1がx軸となす角をarctan2により求める
        beta = np.arctan2(b2, b1)
        
        # l0とl1の交点を求める
        cx, cy = getCrossPointFromPoint(l0_x[0], l0_y[0], l1_x[0], l1_y[0], l0_x[1], l0_y[1], l1_x[1], l1_y[1])
        
        # 幾何より、l0延長線上のフィレット開始点（p1）および、フィレットの中心座標（p0）を求める
        p1_x = cx - r * (1/np.tan(sita)) * np.cos(-alpha)
        p1_y = cy + r * (1/np.tan(sita)) * np.sin(-alpha)

        p2_x = cx - r * (1/np.tan(sita)) * np.cos(-beta)
        p2_y = cy + r * (1/np.tan(sita)) * np.sin(-beta)
        
        m2_0 = np.tan(alpha + np.pi/2)
        m2_1 = np.tan(beta + np.pi/2)
        b0 = -m2_0*p1_x + p1_y
        b1 = -m2_1*p2_x + p2_y
        
        f_x, f_y = getCrossPointFromLines(m2_0, b0, m2_1, b1)

        # 補完点郡の作成に用いる、sitaの配列を作成する。
        sita1 = np.arctan2(p1_y-f_y, p1_x-f_x)
        sita2 = np.arctan2(p2_y-f_y, p2_x-f_x)
        
        sita_array = getFiletSitaArray(sita1, sita2)
        
        # 幾何より、補完点列を作成する
        x_intp = r*np.cos(sita_array) + f_x
        y_intp = r*np.sin(sita_array) + f_y
        
        # l0とl1の端点をフィレットに一致するように調整
        new_l0_x = np.array([l0_x[0], p1_x])
        new_l0_y = np.array([l0_y[0], p1_y])
        new_l1_x = np.array([p2_x, l1_x[1]])
        new_l1_y = np.array([p2_y, l1_y[1]])
              
    except:
        traceback.print_exc()
        #内積・外積が計算できない場合は、補完しない直線を返す
        x_intp = np.array([l0_x[1], l1_x[0]])
        y_intp = np.array([l0_y[1], l1_y[0]])
        new_l0_x = l0_x
        new_l0_y = l0_y
        new_l1_x = l1_x
        new_l1_y = l1_y
        pass
    
    return x_intp, y_intp, new_l0_x, new_l0_y, new_l1_x, new_l1_y, f_x, f_y


def getLineFuncFromPoint(p0_x, p0_y, p1_x, p1_y):
    a = (p1_y - p0_y)/(p1_x - p0_x)
    b = -a*p0_x + p0_y
    
    def line_func(x):
        return a*x+b
    
    return line_func


def getLineFuncFromAB(a, b):
    def line_func(x):
        return a*x+b
    
    return line_func


def getLineFuncFromAPoint(a, p_x, p_y):
    def line_func(x):
        return a*x - a*p_x + p_y
    
    return line_func


def filetLineCurve(a, b, tck, u0, mode, r):
    l_func = getLineFuncFromAB(a, b)
    c_func = getInterpFunc(tck, 0)
    cp_func = getInterpFunc(tck, 1)
    
    u_root, cx, cy = getCrossPointFromCurveLine(l_func, c_func, u0)
    
    if mode == 1:
        fx_est = cx + r*np.cos(np.pi/4)
        fy_est = cy + r*np.sin(np.pi/4)
    elif mode == 2:
        fx_est = cx - r*np.cos(np.pi/4)
        fy_est = cy + r*np.sin(np.pi/4)
    elif mode == 3:
        fx_est = cx - r*np.cos(np.pi/4)
        fy_est = cy - r*np.sin(np.pi/4)
    else:
        fx_est = cx + r*np.cos(np.pi/4)
        fy_est = cy - r*np.sin(np.pi/4)

        
    xu0 = [c_func(u0)[0], u0]
        
    try:
        def calc(xu):
            lx = xu[0]
            cu = xu[1]
            
            ly = l_func(lx)
            lva = -1/a
            lvb = -lva*lx + ly
            
            cp = c_func(cu)
            cx = cp[0]
            cy = cp[1]
            ca = cp_func(cu)
            cva = -1/ca[1] * ca[0]
            cvb = -cva*cx + cy
            
            fx, fy = getCrossPointFromLines(lva, lvb, cva, cvb)
            
            return lx, ly, cx, cy, fx, fy
        
        def solver(xu):
            lx, ly, cx, cy, fx, fy = calc(xu)
            dist_l_f2 = (lx-fx)**2 + (ly-fy)**2
            dist_c_f2 = (cx-fx)**2 + (cy-fy)**2
            
            return (dist_l_f2 - dist_c_f2)**2 + (2*r**2 - dist_l_f2 - dist_c_f2)**2 + ((fx_est-fx)**2 + (fy_est-fy)**2)**2
        
        opt_xu = fmin(solver, x0 = xu0)
        p1_x, p1_y, p2_x, p2_y, f_x, f_y = calc(opt_xu)
        
        r_1 = np.sqrt((p1_x-f_x)**2 + (p1_y-f_y)**2)
        r_2 = np.sqrt((p2_x-f_x)**2 + (p2_y-f_y)**2)
        
        # 補完点郡の作成に用いる、sitaの配列を作成する。
        sita1 = np.arctan2(p1_y-f_y, p1_x-f_x)
        sita2 = np.arctan2(p2_y-f_y, p2_x-f_x)
        
        sita_array = getFiletSitaArray(sita1, sita2)
        
        # 幾何より、補完点列を作成する
        x_intp = r_1*np.cos(sita_array) + f_x
        y_intp = r_2*np.sin(sita_array) + f_y 
              
    except:
        traceback.print_exc()
        opt_xu = xu0
        pass
    
    return opt_xu, p1_x, p1_y, p2_x, p2_y, cx, cy, f_x, f_y, x_intp, y_intp