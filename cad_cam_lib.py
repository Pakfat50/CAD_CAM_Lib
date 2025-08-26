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
N_FILLET_INTERPORATE = 20 # フィレット点数
N_CIRCLE = 100 # 円の生データ点数


class Line:
    def __init__(self, x, y, lineType):
        self.x = np.array(x) 
        self.y = np.array(y)
        self.line_type = lineType
        self.st = np.array([x[0], y[0]])
        self.ed = np.array([x[-1], y[-1]])

class Spline(Line):
    def __init__(self, x, y):
        temp_x, temp_y = removeSamePoint(x, y)
        super().__init__(temp_x, temp_y, "Spline")
        tck, u = intp.splprep([self.x, self.y], k=3, s=0)
        self.tck = tck
        self.u = u
        self.f_curve = getInterpFunc(self.tck, 0)
        self.f_diff = getInterpFunc(self.tck, 1)
        
    def getPoint(self, u):
        p = self.f_curve(u)
        return p[0], p[1]
    
    def getM1(self, u):
        p = self.f_curve(u)
        d = self.f_diff(u)
        return p[0], d[1]/d[0]

    def getM2(self, u):
        p = self.f_curve(u)
        d = self.f_diff(u)
        return p[0], -d[0]/d[1]


class Arc(Spline):
    def __init__(self, x, y, cx, cy):
        temp_x, temp_y = removeSamePoint(x, y)
        super().__init__(temp_x, temp_y) 
        self.line_type = "Arc"
        self.cx = cx
        self.cy = cy
        self.sita_st = np.arctan2(y[0]-cy, x[0]-cx)
        self.sita_ed = np.arctan2(y[-1]-cy, x[-1]-cx)
    
        
class Circle(Spline):
    def __init__(self, r, cx, cy):
        self.sita = np.linspace(0, 2*np.pi, N_CIRCLE)
        x = r*np.cos(self.sita) + cx
        y = r*np.sin(self.sita) + cy
        super().__init__(x, y) 
        self.line_type = "Circle"
        self.cx = cx
        self.cy = cy
        self.r = r


class Ellipse(Spline):
    def __init__(self, a, b, deg, cx, cy):
        rot = np.radians(deg)
        self.sita = np.linspace(0, 2*np.pi, N_CIRCLE)
        x = a*np.cos(rot)*np.cos(self.sita) - b*np.sin(rot)*np.sin(self.sita) + cx
        y = a*np.sin(rot)*np.cos(self.sita) + b*np.cos(rot)*np.sin(self.sita) + cy
        super().__init__(x, y) 
        self.line_type = "Ellipse"
        self.cx = cx
        self.cy = cy
        self.a = a
        self.b = b
        self.deg = deg
        self.rot = rot
        
 
class Polyline(Spline):
    def __init__(self, x, y):
        temp_x, temp_y = removeSamePoint(x, y)
        super().__init__(temp_x, temp_y)
        tck, u = intp.splprep([self.x, self.y], k=1, s=0)
        self.line_type = "Polyiline"
        self.tck = tck
        self.u = u
        self.f_curve = getInterpFunc(self.tck, 0)
        self.f_diff = getInterpFunc(self.tck, 1)
        
        
class SLine(Line):
    def __init__(self, x, y):
        super().__init__(x, y, "SLine")
        self.a = (y[1]-y[0])/(x[1]-x[0])
        self.m1 = self.a
        self.m2 = -1/self.m1
        self.b = -self.a*x[0] + y[0]
        self.sita = np.arctan(self.a)
        self.f_line = getLineFuncFromAB(self.a, self.b)
        

class CLine(Line):
    def __init__(self, x, y):
        super().__init__(x, y, "CLine")

class Airfoil(CLine):
    def __init__(self, x, y):
        super().__init__(x, y, "Airfoils")    

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


def getInterpData(x, y, dim):
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


def getCrossPointFromCurves(curve_func1, curve_func2, u0, s0):
    def solver(us):
        u = us[0]
        s = us[1]
        p1 = curve_func1(u)
        p2 = curve_func2(s)
        err = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        return err
    
    us0 = [u0, s0]
    us_root = fmin(solver, x0 = us0)
    u_root = us_root[0]
    s_root = us_root[1]
    p_root = curve_func1(u_root)
    x_root = p_root[0]
    y_root = p_root[1]
    return u_root, s_root, x_root, y_root


def move(line, mx, my):
    new_x = line.x + mx
    new_y = line.y + my
    
    if line.line_type == "Spline":
        return Spline(new_x, new_y)
    
    elif line.line_type == "Polyline":
        return Polyline(new_x, new_y)
    
    elif line.line_type == "SLine":
        return SLine(new_x, new_y)
    
    elif line.line_type == "Arc":
        return Arc(new_x, new_y, line.cx + mx, line.cy + my)
    
    elif line.line_type == "Circle":
        return Circle(line.r, line.cx + mx, line.cy + my)

    elif line.line_type == "Ellipse":
        return Ellipse(line.a, line.b, line.deg, line.cx + mx, line.cy + my)

def rotate(line, deg, rx, ry):
    
    sita = np.radians(deg)
    
    if (line.line_type == "Arc") or (line.line_type == "Circle") or (line.line_type == "Ellipse") :
        x = line.cx - rx
        y = line.cy - ry
        new_cx = np.cos(sita)*x - np.sin(sita)*y + rx
        new_cy = np.sin(sita)*x + np.cos(sita)*y + ry
        
    new_x = []
    new_y = []
    
    i = 0
    while i < len(line.x):
        x = line.x[i] - rx
        y = line.y[i] - ry

        new_x.append(np.cos(sita)*x - np.sin(sita)*y + rx)
        new_y.append(np.sin(sita)*x + np.cos(sita)*y + ry)
        
        i += 1
        
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    
    if line.line_type == "Spline":
        return Spline(new_x, new_y)
    
    elif line.line_type == "Polyline":
        return Polyline(new_x, new_y)
    
    elif line.line_type == "SLine":
        return SLine(new_x, new_y)
    
    elif line.line_type == "Arc":
        return Arc(new_x, new_y, new_cx, new_cy)

    elif line.line_type == "Circle":
        return Circle(line.r, new_cx, new_cy)

    elif line.line_type == "Ellipse":
        return Ellipse(line.a, line.b, line.deg + deg, new_cx, new_cy)


def offset(line, d):
    if (line.line_type == "SLine"):
        new_x = line.x - d*np.sin(line.sita)
        new_y = line.y + d*np.cos(line.sita)
    else:
        new_x = []
        new_y = []
        
        i = 0
        while i < len(line.x):
            x = line.x[i]
            y = line.y[i]
            u = line.u[i]  #Splineのuは、各x, y座標に対応するuを格納している
            
            # m1を用いてarctanで傾きを求めると、象限の情報が失われてしまうため、微分値からarctan2を用いて求める
            diff = line.f_diff(u)
            sita = np.arctan2(diff[1], diff[0])
            
            new_x.append(x - d*np.sin(sita))
            new_y.append(y + d*np.cos(sita))
            i += 1
            
        new_x = np.array(new_x)
        new_y = np.array(new_y)        
    
    if line.line_type == "Spline":
        return Spline(new_x, new_y)
    
    elif line.line_type == "Polyline":
        return Polyline(new_x, new_y)
    
    elif line.line_type == "SLine":
        return SLine(new_x, new_y)
    
    elif line.line_type == "Arc":
        return Arc(new_x, new_y, line.cx, line.cy)

    elif line.line_type == "Circle":
        return Circle(line.r+d, line.cx, line.cy)

    elif line.line_type == "Ellipse":
        return Ellipse(line.a+d, line.b+d, line.deg, line.cx, line.cy)


def scale(line, s, x0, y0):
    
    new_x = (line.x - x0) * s + x0
    new_y = (line.y - y0) * s + y0
 
    if (line.line_type == "Arc") or (line.line_type == "Circle") or (line.line_type == "Ellipse") :
        new_cx = (line.cx - x0) * s + x0
        new_cy = (line.cy - y0) * s + y0
        
    if line.line_type == "Spline":
        return Spline(new_x, new_y)
    
    elif line.line_type == "Polyline":
        return Polyline(new_x, new_y)
    
    elif line.line_type == "SLine":
        return SLine(new_x, new_y)
    
    elif line.line_type == "Arc":
        return Arc(new_x, new_y, new_cx, new_cy)

    elif line.line_type == "Circle":
        return Circle(line.r*s, new_cx, new_cy)

    elif line.line_type == "Ellipse":
        return Ellipse(line.a*s, line.b*s, line.deg, new_cx, new_cy)



def getFiletSitaArray(sita1, sita2):
    if sita2-sita1 <= np.pi and sita2-sita1 > -np.pi:
        sita_array = np.linspace(sita1, sita2, N_FILLET_INTERPORATE)
    elif sita2-sita1 > np.pi:
        sita_array = np.linspace(sita1, sita2-2*np.pi, N_FILLET_INTERPORATE)
    else:
        sita_array = np.linspace(sita1, sita2+2*np.pi, N_FILLET_INTERPORATE)
    return sita_array   


def filetLines(l0, l1, r):
    try:
        # l0とl1のなす角sitaを内積により求める
        # https://w3e.kanazawa-it.ac.jp/math/category/vector/henkan-tex.cgi?target=/math/category/vector/naiseki-wo-fukumu-kihonsiki.html&pcview=2
        a1 = l0.x[1] - l0.x[0]
        a2 = l0.y[1] - l0.y[0]
        b1 = l1.x[0] - l1.x[1]
        b2 = l1.y[0] - l1.y[1]
            
        sita = np.arccos( (a1*b1 + a2*b2)/(np.sqrt(a1**2 + a2**2) * np.sqrt(b1**2 + b2**2)) ) / 2.0

        # l0がx軸となす角をarctan2により求める
        alpha = np.arctan2(a2, a1)
        
        # l1がx軸となす角をarctan2により求める
        beta = np.arctan2(b2, b1)
        
        # l0とl1の交点を求める
        cx, cy = getCrossPointFromPoint(l0.x[0], l0.y[0], l1.x[0], l1.y[0], l0.x[1], l0.y[1], l1.x[1], l1.y[1])
        
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
        filet = Arc(x_intp, y_intp, f_x, f_y)
        
        # l0とl1の端点をフィレットに一致するように調整
        new_l0 = SLine([l0.x[0], p1_x], [l0.y[0], p1_y])
        new_l1 = SLine([p2_x, l1.x[1]], [p2_y, l1.y[1]])
              
    except:
        traceback.print_exc()
        pass
    
    return new_l0, new_l1, filet


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


def ellipseAB(x1, y1, x2, y2, ox, oy):
    a = (x1-ox)**2
    b = (y1-oy)**2
    c = (x2-ox)**2
    d = (y2-oy)**2
    detA = a*d-b*c
    A = 1/np.sqrt((d-b)/detA)
    B = 1/np.sqrt((a-c)/detA)
    
    return A, B

def filetLineCurve(line, spline, r, mode, u0):
    u_root, cx, cy = getCrossPointFromCurveLine(line.f_line, spline.f_curve, u0)
    cp_root = spline.f_diff(u_root)
    
    sita_c = np.arctan(cp_root[1]/cp_root[0])
    sita_l = line.sita
    sita_min = min(sita_c, sita_l)
    sita_max = max(sita_c, sita_l)
    
    if mode == 1:
        sita = np.abs((sita_l-sita_c)/2)
        
        r_f = np.abs(r/np.sin(sita))
        fx_est = cx + r_f * np.cos(sita_min+sita)
        fy_est = cy + r_f * np.sin(sita_min+sita)
    elif mode == 2:
        sita = np.abs((np.pi-sita_l+sita_c)/2)
        r_f = np.abs(r/np.sin(sita))
        fx_est = cx + r_f * np.cos(sita_max+sita)
        fy_est = cy + r_f * np.sin(sita_max+sita)
    elif mode == 3:
        sita = np.abs((sita_l-sita_c)/2)
        r_f = np.abs(r/np.sin(sita))
        fx_est = cx + r_f * np.cos(sita_min+sita+np.pi)
        fy_est = cy + r_f * np.sin(sita_min+sita+np.pi)
    else:
        sita = np.abs((np.pi-sita_l+sita_c)/2)
        r_f = np.abs(r/np.sin(sita))
        fx_est = cx + r_f * np.cos(sita_max+sita+np.pi)
        fy_est = cy + r_f * np.sin(sita_max+sita+np.pi)

    xu0 = [spline.f_curve(u0)[0], u0]
        
    try:
        def calc(xu):
            lx = xu[0]
            cu = xu[1]
            
            ly = line.f_line(lx)
            lva = line.m2
            lvb = -lva*lx + ly
            
            cp = spline.f_curve(cu)
            cx = cp[0]
            cy = cp[1]
            ca = spline.f_diff(cu)
            cva = -1/ca[1] * ca[0]
            cvb = -cva*cx + cy
            
            fx, fy = getCrossPointFromLines(lva, lvb, cva, cvb)
            
            return lx, ly, cx, cy, fx, fy
        
        def rough_solver(xu):
            lx, ly, cx, cy, fx, fy = calc(xu)
            dist_l_f2 = (lx-fx)**2 + (ly-fy)**2
            dist_c_f2 = (cx-fx)**2 + (cy-fy)**2
            
            return (dist_l_f2 - dist_c_f2)**2 + (2*r**2 - dist_l_f2 - dist_c_f2)**2 + ((fx_est-fx)**2 + (fy_est-fy)**2)**2

        def solver(xu):
            lx, ly, cx, cy, fx, fy = calc(xu)
            dist_l_f2 = (lx-fx)**2 + (ly-fy)**2
            dist_c_f2 = (cx-fx)**2 + (cy-fy)**2
            
            return (dist_l_f2 - dist_c_f2)**2 + (2*r**2 - dist_l_f2 - dist_c_f2)**2
        
        rough_xu = fmin(rough_solver, x0 = xu0)
        opt_xu = fmin(rough_solver, x0 = rough_xu)
        
        p1_x, p1_y, p2_x, p2_y, f_x, f_y = calc(opt_xu)
        
        a,b = ellipseAB(p1_x, p1_y, p2_x, p2_y, f_x, f_y)
        
        # 補完点郡の作成に用いる、sitaの配列を作成する。
        sita1 = np.arctan2(p1_y-f_y, p1_x-f_x)
        sita2 = np.arctan2(p2_y-f_y, p2_x-f_x)
        sita_array = getFiletSitaArray(sita1, sita2)
        
        # 補完点列を作成する
        x_intp = a*np.cos(sita_array) + f_x
        y_intp = b*np.sin(sita_array) + f_y 
              
    except:
        traceback.print_exc()
        pass
    
    return opt_xu, p1_x, p1_y, p2_x, p2_y, cx, cy, f_x, f_y, x_intp, y_intp, fx_est, fy_est

def filetCurves(spline1, spline2, r, mode, u0, s0):
    u_root, s_root, cx, cy = getCrossPointFromCurves(spline1.f_curve, spline2.f_curve, u0, s0)
    cp1_root = spline1.f_diff(u_root)
    cp2_root = spline2.f_diff(s_root)
    
    sita1 = np.arctan(cp1_root[1]/cp1_root[0])
    sita2 = np.arctan(cp2_root[1]/cp2_root[0])
    sita_min = min(sita1, sita2)
    sita_max = max(sita1, sita2)
    
    if mode == 1:
        sita = np.abs((sita1-sita2)/2)
        
        r_f = np.abs(r/np.sin(sita))
        fx_est = cx + r_f * np.cos(sita_min+sita)
        fy_est = cy + r_f * np.sin(sita_min+sita)
    elif mode == 2:
        sita = np.abs((np.pi-sita1+sita2)/2)
        r_f = np.abs(r/np.sin(sita))
        fx_est = cx + r_f * np.cos(sita_max+sita)
        fy_est = cy + r_f * np.sin(sita_max+sita)
    elif mode == 3:
        sita = np.abs((sita1-sita2)/2)
        r_f = np.abs(r/np.sin(sita))
        fx_est = cx + r_f * np.cos(sita_min+sita+np.pi)
        fy_est = cy + r_f * np.sin(sita_min+sita+np.pi)
    else:
        sita = np.abs((np.pi-sita1+sita2)/2)
        r_f = np.abs(r/np.sin(sita))
        fx_est = cx + r_f * np.cos(sita_max+sita+np.pi)
        fy_est = cy + r_f * np.sin(sita_max+sita+np.pi)

    us0 = [u0, s0]
        
    try:
        def calc(us):
            u = us[0]
            s = us[1]
            
            x1,y1 = spline1.getPoint(u)
            x2,y2 = spline2.getPoint(s)
            a1 = spline1.f_diff(u)
            a2 = spline2.f_diff(s)
            va1 = -1/a1[1] * a1[0]
            va2 = -1/a2[1] * a2[0]
            vb1 = -va1*x1 + y1
            vb2 = -va2*x2 + y2
            
            fx, fy = getCrossPointFromLines(va1, vb1, va2, vb2)
            
            return x1, y1, x2, y2, fx, fy
        
        def rough_solver(us):
            lx, ly, cx, cy, fx, fy = calc(us)
            dist_l_f2 = (lx-fx)**2 + (ly-fy)**2
            dist_c_f2 = (cx-fx)**2 + (cy-fy)**2
            
            return (dist_l_f2 - dist_c_f2)**2 + (2*r**2 - dist_l_f2 - dist_c_f2)**2 + ((fx_est-fx)**2 + (fy_est-fy)**2)**2

        def solver(us):
            lx, ly, cx, cy, fx, fy = calc(us)
            dist_l_f2 = (lx-fx)**2 + (ly-fy)**2
            dist_c_f2 = (cx-fx)**2 + (cy-fy)**2
            
            return (dist_l_f2 - dist_c_f2)**2 + (2*r**2 - dist_l_f2 - dist_c_f2)**2
        
        rough_xu = fmin(rough_solver, x0 = us0)
        opt_xu = fmin(rough_solver, x0 = rough_xu)
        
        p1_x, p1_y, p2_x, p2_y, f_x, f_y = calc(opt_xu)
        
        a,b = ellipseAB(p1_x, p1_y, p2_x, p2_y, f_x, f_y)
        
        # 補完点郡の作成に用いる、sitaの配列を作成する。
        sita1 = np.arctan2(p1_y-f_y, p1_x-f_x)
        sita2 = np.arctan2(p2_y-f_y, p2_x-f_x)
        sita_array = getFiletSitaArray(sita1, sita2)
        
        # 補完点列を作成する
        x_intp = a*np.cos(sita_array) + f_x
        y_intp = b*np.sin(sita_array) + f_y 
              
    except:
        traceback.print_exc()
        pass
    
    return opt_xu, p1_x, p1_y, p2_x, p2_y, cx, cy, f_x, f_y, x_intp, y_intp, fx_est, fy_est


def trim(line, st, ed):

    if (line.line_type == "SLine"):
        x_st = st
        x_ed = ed
        new_x = np.array([x_st, x_ed])
        new_y = line.f_line(new_x)
        
    if (line.line_type == "Spline") or (line.line_type == "Polyline") or (line.line_type == "Arc"):
        u_st = st
        u_ed = ed
        
        x_st, y_st = line.getPoint(u_st)
        x_ed, y_ed = line.getPoint(u_ed)
        new_x = [x_st]
        new_y = [y_st]
        
        i = 0
        while i < len(line.u):
            x = line.x[i]
            y = line.y[i]
            u = line.u[i]  #Splineのuは、各x, y座標に対応するuを格納している
            
            if (u >= u_st) and (u <= u_ed):
                new_x.append(x)
                new_y.append(y)
            i += 1
        new_x.append(x_ed)
        new_y.append(y_ed)    
        
        new_x = np.array(new_x)
        new_y = np.array(new_y)        
    
    if line.line_type == "Circle":
        sita_st = st
        sita_ed = ed
        sita = np.linspace(sita_st, sita_ed, N_CIRCLE)
        new_x = line.r*np.cos(sita) + line.cx
        new_y = line.r*np.sin(sita) + line.cy
        
    if line.line_type == "Ellipse":
        sita_st = st
        sita_ed = ed
        sita = np.linspace(sita_st, sita_ed, N_CIRCLE)    
        new_x = line.a*np.cos(line.rot)*np.cos(sita) - line.b*np.sin(line.rot)*np.sin(sita) + line.cx
        new_y = line.a*np.sin(line.rot)*np.cos(sita) + line.b*np.cos(line.rot)*np.sin(sita) + line.cy
        
        
    if line.line_type == "Spline":
        return Spline(new_x, new_y)
    
    elif line.line_type == "Polyline":
        return Polyline(new_x, new_y)
    
    elif line.line_type == "SLine":
        return SLine(new_x, new_y)
    
    elif line.line_type == "Arc":
        return Arc(new_x, new_y, line.cx, line.cy)

    elif line.line_type == "Circle":
        #円をトリムしたオブジェクトは円弧で返す
        return Arc(new_x, new_y, line.cx, line.cy)

    elif line.line_type == "Ellipse":
        #楕円をトリムしたオブジェクトはスプラインで返す
        return Spline(new_x, new_y)
    
    
    



