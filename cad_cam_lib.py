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
from shapely.geometry import Point, Polygon

# グローバル変数
N_FILLET_INTERPORATE = 20 # フィレット点数
N_CIRCLE = 100 # 円の生データ点数
N_AIRFOIL = 500 # 翼形のデータ数
N_AIRFOIL_INTERPOLATE = 1000 # 翼形補完時の補完点数
DELTA_U = 0.01 # ベクトル算出用のu差分
DIST_NEAR = 0.00001 # 近傍点の判定距離
DXF_LINETYPES_DEFAULT = "ByLayer" #dxfファイルのデフォルト線種
DXF_COLOR_DEFAULT = 0 # dxfファイルのデフォルト線色。AutoCAD Color Index (ACI)で指定
DXF_WIDTH_DEFAULT = 2 # dxfファイルのデフォルト線幅。
DXF_USE_SPLINE = True # dxfの出力でスプラインをスプラインとして出力する


class Line:
    def __init__(self, x, y, lineType):
        self.x = np.array(x) 
        self.y = np.array(y)
        self.line_type = lineType
        self.st = np.array([x[0], y[0]])
        self.ed = np.array([x[-1], y[-1]])
        self.length = getLength(x, y)


class SLine(Line):
    def __init__(self, x, y):
        super().__init__(x, y, "SLine")
        if x[1] == x[0]:
            self.a = np.inf
        else: 
            self.a = (y[1]-y[0])/(x[1]-x[0])
        
        self.m1 = self.a
        
        if self.m1 == 0:
            self.m2 = np.inf
        else:
            self.m2 = -1/self.m1
            
        self.b = -self.a*x[0] + y[0]
        self.sita = np.arctan2(y[1]-y[0], x[1]-x[0])
        self.f_line = getLineFuncFromAB(self.a, self.b)
        

class Spline(Line):
    def __init__(self, x, y):
        temp_x, temp_y = removeSamePoint(x, y)
        super().__init__(temp_x, temp_y, "Spline")
        tck, u = intp.splprep([self.x, self.y], k=3, s=0)
        self.tck = tck
        self.u = u
        self.f_curve = getInterpFunc(self.tck, 0)
        self.f_diff = getInterpFunc(self.tck, 1)
        self.x_intp = self.x
        self.y_intp = self.y
        
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
    
    def getSita1(self, u):
        d = self.f_diff(u)
        return np.arctan2(d[1], d[0])
    
    def getSita2(self, u):
        d = self.f_diff(u)
        return np.arctan2(-d[0], d[1])

    def setIntporatePoints(self, u):
        temp_x, temp_y = self.getPoint(u)
        self.x_intp = temp_x
        self.y_intp = temp_y
        
    def getUfromX(self, x):
        if (x >= min(self.x)) and (x <= max(self.x)) :
            tck = intp.splrep(self.u, self.x - x, k = 3)
            u_root = intp.sproot(tck)
            return u_root
        else:
            return []
        
    def getUfromY(self, y):
        if (y >= min(self.y)) and (y <= max(self.y)) :
            tck = intp.splrep(self.u, self.y - y, k = 3)
            u_root = intp.sproot(tck)
            return u_root
        else:
            return []

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
        

class Arc(Spline):
    def __init__(self, r, cx, cy, sita_st, sita_ed):
        if r <= DIST_NEAR:
            r = DIST_NEAR
            N = 4
        else:
            N = N_CIRCLE
        self.sita_st = sita_st
        self.sita_ed = sita_ed
        self.sita = np.linspace(self.sita_st, self.sita_ed, N)
        self.r = r
        self.cx = cx
        self.cy = cy

        x = r*np.cos(self.sita) + cx
        y = r*np.sin(self.sita) + cy
        
        super().__init__(x, y) 
        self.line_type = "Arc"


class EllipseArc(Spline):
    def __init__(self, a, b, rot, cx, cy, sita_st, sita_ed):
        if (a <= DIST_NEAR) or (b <= DIST_NEAR):
            a = DIST_NEAR
            b = DIST_NEAR
            N = 4
        else:
            N = N_CIRCLE
        self.sita_st = sita_st
        self.sita_ed = sita_ed
        self.sita = np.linspace(self.sita_st, self.sita_ed, N)
        self.rot = rot
        self.a = a
        self.b = b
        self.cx = cx
        self.cy = cy
        
        x = a*np.cos(self.rot)*np.cos(self.sita) - b*np.sin(self.rot)*np.sin(self.sita) + cx
        y = a*np.sin(self.rot)*np.cos(self.sita) + b*np.cos(self.rot)*np.sin(self.sita) + cy
        
        super().__init__(x, y) 
        self.line_type = "EllipseArc"


class Circle(Arc):
    def __init__(self, r, cx, cy, ccw = True):         
        if ccw == True:
            super().__init__(r, cx, cy, 0, 2*np.pi) 
        else:
            super().__init__(r, cx, cy, 2*np.pi, 0) 
        self.line_type = "Circle"
        self.area = np.pi * self.r**2
        self.ccw = ccw


class Ellipse(EllipseArc):
    def __init__(self, a, b, rot, cx, cy, ccw = True):
        if ccw == True:
            super().__init__(a, b, rot, cx, cy, 0, 2*np.pi) 
        else:
            super().__init__(a, b, rot, cx, cy, 2*np.pi, 0) 
        self.line_type = "Ellipse"
        self.area = np.pi * self.a * self.b
        self.ccw = ccw
        

class Airfoil(Spline):
    def __init__(self, x, y):
        # 重複した座標点を削除
        temp_x, temp_y = removeSamePoint(x, y)
        
        # スプラインを初期化
        super().__init__(temp_x, temp_y)
        self.line_type = "Airfoil"
        
        self.u_zero = self.u[np.argmin(self.x)]
        u_intp = getUCosine(N_AIRFOIL_INTERPOLATE, self.u_zero)
        self.setIntporatePoints(u_intp)
        
        # 翼形が時計回りか反時計回りかを検出
        self.ccw = detectRotation(self.x_intp, self.y_intp)
        zero_index = np.argmin(self.x_intp)
        if self.ccw == True:
            # 反時計回り      
            self.ux = self.x_intp[:zero_index+1][-1::-1]
            self.uy = self.y_intp[:zero_index+1][-1::-1]
            self.lx = self.x_intp[zero_index:]
            self.ly = self.y_intp[zero_index:]
        else:
            # 時計回り
            self.lx = self.x_intp[:zero_index+1][-1::-1]
            self.ly = self.y_intp[:zero_index+1][-1::-1]
            self.ux = self.x_intp[zero_index:]
            self.uy = self.y_intp[zero_index:]           
        
        self.xmin = max(min(self.ux), min(self.lx))
        self.xmax = min(max(self.ux), max(self.lx))
        self.f_upper = intp.interp1d(self.ux, self.uy, kind = 'cubic')
        self.f_lower = intp.interp1d(self.lx, self.ly, kind = 'cubic')
        cx = getXCosine(self.xmin, self.xmax, int(N_AIRFOIL_INTERPOLATE/2))
        cy = (self.f_upper(cx) + self.f_lower(cx))/2.0
        self.f_center = intp.interp1d(cx, cy, kind = 'cubic')


class LineGroup:
    def __init__(self, line_list):
        self.lines = line_list
        self.line_type = "LineGroup"
        self.d = 0
        self.update()
        
    def update(self):
        x = np.array([])
        y = np.array([])
        
        for line in self.lines:
            x = np.concatenate([x, line.x], 0)
            y = np.concatenate([y, line.y], 0)
        
        self.x = x
        self.y = y
        self.st = np.array([x[0], y[0]])
        self.ed = np.array([x[-1], y[-1]])
        self.length = getLength(x, y)
        self.ccw = detectRotation(self.x, self.y)
        self.closed = checkIsClosed(self.st[0], self.st[1], self.ed[0], self.ed[1])
        
    def invert(self, num):
        if num < len(self.lines):
            line = self.lines[num]
            invert_line = invert(line)
            self.lines[num] = invert_line
            self.update()
    
    def invertAll(self):
        i = 0
        while i < len(self.lines):
            line = self.lines[i]
            invert_line = invert(line)
            self.lines[i] = invert_line
            i += 1
        self.lines.reverse()
        self.update()
            
    def sort(self, start_num):
        connect_lines = sortLines(self.lines, start_num)
        self.lines = connect_lines
        self.update()
        
    def offset(self, d):
        lines = []
        for line in self.lines:
            # オフセットする曲線が、自己交差を持ちうる場合、自己交差除去を行う
            # 自己交差を持ちうるのはスプライン、ポリライン、翼形のみ
            if (line.line_type == "Spline") or (line.line_type == "Polyline") or \
                (line.line_type == "Airfoil"):
                o_line = offset(line, d, True)
            else:
                o_line = offset(line, d)
            lines.append(o_line)
        self.lines = lines
        self.update()
        self.d = d
        
    def insertFilet(self):
        if np.abs(self.d) > DIST_NEAR:
            lines = [self.lines[0]]
            i = 1
            while i < len(self.lines):
                line_st = self.lines[i-1]
                line_ed = self.lines[i]
                sl_st = SLine([line_st.x[-2], line_st.x[-1]], [line_st.y[-2], line_st.y[-1]])
                sl_ed = SLine([line_ed.x[0], line_ed.x[1]], [line_ed.y[0], line_ed.y[1]])
                new_l0, new_l1, filet = filetLines(sl_st, sl_ed, np.abs(self.d))
                lines.append(filet)
                lines.append(self.lines[i])
                i += 1
                
            line_st = self.lines[-1]
            line_ed = self.lines[0]
            sl_st = SLine([line_st.x[-2], line_st.x[-1]], [line_st.y[-2], line_st.y[-1]])
            sl_ed = SLine([line_ed.x[0], line_ed.x[1]], [line_ed.y[0], line_ed.y[1]])
            new_l0, new_l1, filet = filetLines(sl_st, sl_ed, np.abs(self.d))
            lines.append(filet)
                
            self.lines = lines
        self.update()
        

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


def norm(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def getLength(x, y):
    i = 0
    length = [0]
    while i < len(x) - 1:
        length.append(norm(x[i], y[i], x[i+1], y[i+1]))
        i += 1
    return np.array(length)


def getArea(x, y):
    if norm(x[0], y[0], x[-1], y[-1]) < DIST_NEAR:
        i = 0
        temp_s = 0
        while i < len(x) - 1:
            temp_s += x[i]*y[i+1] - x[i+1]*y[i]
            i += 1
        temp_s += x[-1]*y[0] - x[0]*y[-1]
        S = 0.5 * np.abs(temp_s)
        return S
    else:
        return 0


def removeSamePoint(x, y):
    i = 0
    new_x = []
    new_y = []
    while i < len(x)-1:
        if not ((x[i+1] == x[i]) and (y[i+1] == y[i])):
            new_x.append(x[i])
            new_y.append(y[i])
        i += 1
    
    new_x.append(x[-1])
    new_y.append(y[-1])
    
    return np.array(new_x), np.array(new_y)


def getNormalizedSumArray(array):
    i = 0
    sum_array = [0]
    temp_sum = 0
    while i < len(array)-1:
        temp_sum += np.abs(array[i+1] - array[i])
        sum_array.append(temp_sum)
        i += 1
    sum_array /= temp_sum
    return sum_array


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


def getInterpFunc(tck, der):
    def interpFunc(u):
        return intp.splev(u, tck, der)
    return interpFunc


def getInterpData(x, y, dim):
    temp_x, temp_y = removeSamePoint(x, y)
    
    return intp.splprep([temp_x, temp_y],k=dim,s=0)


def invert(line):
    new_x = line.x[-1::-1]
    new_y = line.y[-1::-1]
    
    if line.line_type == "Spline":
        return Spline(new_x, new_y)
    
    elif line.line_type == "Polyline":
        return Polyline(new_x, new_y)
    
    elif line.line_type == "SLine":
        return SLine(new_x, new_y)
    
    elif line.line_type == "Arc":
        return Arc(line.r, line.cx, line.cy, line.sita_ed, line.sita_st)

    elif line.line_type == "Circle":
        return Circle(line.r, line.cx, line.cy, not(line.ccw))

    elif line.line_type == "Ellipse":
        return Ellipse(line.a, line.b, line.rot, line.cx, line.cy, not(line.ccw))

    elif line.line_type == "Airfoil":
        return Airfoil(new_x, new_y)

    elif line.line_type == "LineGroup":
        return line.invertAll()    

def detectRotation(x, y):
    i = 0
    temp_s = 0

    while i < len(x) - 1:
        temp_s += x[i]*y[i+1] - x[i+1]*y[i]
        i += 1
    temp_s += x[-1]*y[0] - x[0]*y[-1]
    
    if temp_s > 0:
        ccw = True
    else:
        ccw = False
    
    return ccw


def checkIsClosed(x_st, y_st, x_ed, y_ed):
    if norm(x_st, y_st, x_ed, y_ed) < DIST_NEAR:
        return True
    else:
        return False



def getUCosine(num, u_c):
    sita1 = np.linspace(0, np.pi/2, int(num/2))
    sita2 = np.linspace(np.pi/2, 0, int(num/2)+1)
    u1 = np.sin(sita1) * u_c
    u2 = 1 - np.sin(sita2) * (1-u_c)
    
    u = np.concatenate([u1, u2[1:]], 0)
    
    return u


def getXCosine(xmin, xmax, num):
    sita = np.linspace(0, np.pi/2, num)
    x = (xmax-xmin)*(1-np.cos(sita)) + xmin
    return x

    
def mixAirfoil(airfoil1, airfoil2, ratio):
    xmin = max(airfoil1.xmin, airfoil2.xmin)
    xmax = min(airfoil1.xmax, airfoil2.xmax)
    x = getXCosine(xmin, xmax, int(N_AIRFOIL/2))
    uy = ratio*airfoil1.f_upper(x) + (1-ratio)*airfoil2.f_upper(x)
    ly = ratio*airfoil1.f_lower(x) + (1-ratio)*airfoil2.f_lower(x)
    
    new_x = np.concatenate([x[-1::-1], x], 0)
    new_y = np.concatenate([uy[-1::-1], ly], 0)
    
    new_airfoil = Airfoil(new_x, new_y)
    
    return new_airfoil


def importFromText(fileName, delimiter, skip_header):
    text_file = np.genfromtxt(fileName, delimiter = delimiter, skip_header = skip_header, dtype = float)
    temp_x = text_file[:,0]
    temp_y = text_file[:,1]
    return temp_x, temp_y


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
        return Arc(line.r, line.cx + mx, line.cy + my, line.sita_st, line.sita_ed)
    
    elif line.line_type == "EllipseArc":
        return Arc(line.a, line.b, line.rot, line.cx + mx, line.cy + my, line.sita_st, line.sita_ed)
        
    elif line.line_type == "Circle":
        return Circle(line.r, line.cx + mx, line.cy + my)

    elif line.line_type == "Ellipse":
        return Ellipse(line.a, line.b, line.rot, line.cx + mx, line.cy + my)

    elif line.line_type == "Airfoil":
        return Airfoil(new_x, new_y)


def rotate(line, sita, rx, ry):
    
    if (line.line_type == "Arc") or (line.line_type == "EllipseArc") or \
        (line.line_type == "Circle") or (line.line_type == "Ellipse") :
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
        return Arc(line.r, new_cx, new_cy, line.sita_st + sita, line.sita_ed + sita)
    
    elif line.line_type == "EllipseArc":
        return Arc(line.a, line.b, line.rot + sita, new_cx, new_cy, line.sita_st, line.sita_ed)

    elif line.line_type == "Circle":
        return Circle(line.r, new_cx, new_cy)

    elif line.line_type == "Ellipse":
        return Ellipse(line.a, line.b, line.rot + sita, new_cx, new_cy)

    elif line.line_type == "Airfoil":
        return Airfoil(new_x, new_y)


def offset(line, d, removeSelfCollision = False):
    # 半径を変化させてオフセットする円系は、回転方向によるオフセット方向の変化をスプライン等とそろえるため、
    # 反時計回りの場合は、オフセット方向を反転させる
    if (line.line_type == "Arc") or (line.line_type == "EllipseArc"):
        if line.sita_st < line.sita_ed: # 反時計回り
            d = -d
    elif (line.line_type == "Circle") or (line.line_type == "Ellipse"):
        if line.ccw == True: # 反時計回り
            d = -d
        
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
        
        if removeSelfCollision == True:
            new_x, new_y = detectSelfCollision(new_x, new_y)
        else:
            new_x = np.array(new_x)
            new_y = np.array(new_y)
        
    
    if line.line_type == "Spline":
        return Spline(new_x, new_y)
    
    elif line.line_type == "Polyline":
        return Polyline(new_x, new_y)
    
    elif line.line_type == "SLine":
        return SLine(new_x, new_y)
    
    elif line.line_type == "Arc":
        return Arc(line.r+d, line.cx, line.cy, line.sita_st, line.sita_ed)
    
    elif line.line_type == "EllipseArc":
        return Arc(line.a+d, line.b+d, line.rot, line.cx, line.cy, line.sita_st, line.sita_ed)

    elif line.line_type == "Circle":
        return Circle(line.r+d, line.cx, line.cy)

    elif line.line_type == "Ellipse":
        return Ellipse(line.a+d, line.b+d, line.rot, line.cx, line.cy)

    elif line.line_type == "Airfoil":
        return Airfoil(new_x, new_y)


def scale(line, s, x0, y0):
    
    new_x = (line.x - x0) * s + x0
    new_y = (line.y - y0) * s + y0
 
    if (line.line_type == "Arc") or (line.line_type == "EllipseArc") or \
        (line.line_type == "Circle") or (line.line_type == "Ellipse") :
        new_cx = (line.cx - x0) * s + x0
        new_cy = (line.cy - y0) * s + y0
        
    if line.line_type == "Spline":
        return Spline(new_x, new_y)
    
    elif line.line_type == "Polyline":
        return Polyline(new_x, new_y)
    
    elif line.line_type == "SLine":
        return SLine(new_x, new_y)
    
    elif line.line_type == "Arc":
        return Arc(line.r*s, new_cx, new_cy, line.sita_st, line.sita_ed)
    
    elif line.line_type == "EllipseArc":
        return Arc(line.a*s, line.b*s, line.rot, new_cx, new_cy, line.sita_st, line.sita_ed)

    elif line.line_type == "Circle":
        return Circle(line.r*s, new_cx, new_cy)

    elif line.line_type == "Ellipse":
        return Ellipse(line.a*s, line.b*s, line.rot, new_cx, new_cy)
    
    elif line.line_type == "Airfoil":
        return Airfoil(new_x, new_y)


def getFiletSita(sita_st, sita_ed):
    
    # 開始角から終了角までの変化量（sita_stとsita_edの傾きをもつ線のなす角）を計算
    delta_sita = sita_ed - sita_st
    
    if delta_sita <= np.pi and delta_sita > -np.pi:
        # 鋭角なのでそのままでOK
        return sita_st, sita_ed
    elif delta_sita > np.pi:
        # 反時計回りとしたとき、終了角が1周分オーバーラップしているので、終了角から2piを引く
        return sita_st, sita_ed-2*np.pi
    else:
        # 反時計回りとしたとき、終了角が1周分不足しているので、終了角に2piを足す
        return sita_st, sita_ed +2*np.pi,


def filetLines(l0, l1, r, join=False):
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

        # 円弧の始点角と終点角を計算する。
        sita_st = np.arctan2(p1_y-f_y, p1_x-f_x)
        sita_ed = np.arctan2(p2_y-f_y, p2_x-f_x)
        sita_st, sita_ed = getFiletSita(sita_st, sita_ed)
        
        # フィレットを円弧で作成する
        filet = Arc(r, f_x, f_y, sita_st, sita_ed)
                
        # l0とl1の端点をフィレットに一致するように調整
        new_l0 = SLine([l0.x[0], p1_x], [l0.y[0], p1_y])
        new_l1 = SLine([p2_x, l1.x[1]], [p2_y, l1.y[1]])
        
        if join == True:
            x = np.append(l0.x[0], filet.x)
            x = np.append(x, l1.x[1])
            y = np.append(l0.y[0], filet.y)
            y = np.append(y, l1.y[1])
            filet = Spline(x, y)

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
        
        # 円弧の始点角と終点角を計算する。
        sita_st = np.arctan2(p1_y-f_y, p1_x-f_x) # lineとの接点
        sita_ed = np.arctan2(p2_y-f_y, p2_x-f_x) # splineとの接点
        sita_st, sita_ed = getFiletSita(sita_st, sita_ed) # sita_st ~ sita_ed が必ず鋭角となるように設定
        
        # 楕円弧を用いてフィレットを作成する
        filet = EllipseArc(a, b, 0, f_x, f_y, sita_st, sita_ed)
        
        # 端点をトリムする        
        # 直線の始点から終点に向かうベクトルを算出
        vect_line = line.ed - line.st
        # フィレットの直線との接点におけるuが増加する方向のベクトルを算出
        vect_filet_l = np.array(filet.f_curve(filet.u[0] + DELTA_U)) - np.array(filet.f_curve(filet.u[0]))        
        
        opt_u = opt_xu[1]
        # スプラインのフィレットとの接点におけるuが増加する方向のベクトルを算出
        vect_spline = np.array(spline.f_curve(opt_u + DELTA_U)) - np.array(spline.f_curve(opt_u))
        # フィレットのスプラインとの接点におけるuが増加する方向のベクトルを算出
        vect_filet_s = np.array(filet.f_curve(filet.u[-1])) - np.array(filet.f_curve(filet.u[-1] - DELTA_U))
                
        # 直線　→　フィレット　→　スプライン　の順にフィレットの点列は生成される
        # よって直線との接点では、フィレットと直線とのベクトルは反対を向き、スプラインとの接点では同じ方向を向く
        if np.dot(vect_line, vect_filet_l) < 0:
            # 直線とフィレットのベクトルが反対方向なので、直線の向きはもとの方向に増加でOK
            # 始点を接点のx,終点を直線の終点としてトリム
            t_line = trim(line, p1_x, line.ed[0])
        else:
            # 直線とフィレットのベクトルが同じ方向なので、直線の向きはもとの方向に増加するとNG
            # 始点を直線の始点、終点を接点のxとしてトリム
            t_line = trim(line, line.st[0], p1_x)        
        
        if np.dot(vect_spline, vect_filet_s) >= 0:
            # スプラインとフィレットのベクトルが同じ方向なので、スプラインはu増加でOK
            # 始点を接点のu、終点をスプライン終点のuとしてトリム
            t_spline = trim(spline, opt_u, spline.u[-1])
        else:
            # スプラインとフィレットのベクトルが逆向きなので、スプラインのuだとNG
            # スプライン始点のu、終点を接点のuとしてトリム
            t_spline = trim(spline, spline.u[0], opt_u)
        
        """
        #for debug
        plt.plot(p1_x, p1_y, "ro")
        plt.plot(p2_x, p2_y, "ko")
        plt.plot(cx, cy, "ko")
        plt.plot(f_x, f_y, "ro")
        plt.plot(fx_est, fy_est, "bo")
        """
        
    except:
        traceback.print_exc()
        pass
    
    return t_line, t_spline, filet


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
        
        rough_us = fmin(rough_solver, x0 = us0)
        opt_us = fmin(rough_solver, x0 = rough_us)
        
        p1_x, p1_y, p2_x, p2_y, f_x, f_y = calc(opt_us)
        
        a,b = ellipseAB(p1_x, p1_y, p2_x, p2_y, f_x, f_y)
        
        # 補完点郡の作成に用いる、sitaの配列を作成する。
        sita_st = np.arctan2(p1_y-f_y, p1_x-f_x) # spline1との接点
        sita_ed = np.arctan2(p2_y-f_y, p2_x-f_x) # spline2との接点
        sita_st, sita_ed = getFiletSita(sita_st, sita_ed) # sita_st ~ sita_ed が必ず鋭角となるように設定
          
        # 楕円弧を用いてフィレットを作成する
        filet = EllipseArc(a, b, 0, f_x, f_y, sita_st, sita_ed)
        
        # 端点をトリムする        
        # スプライン1のフィレットとの接点におけるuが増加する方向のベクトルを算出
        opt_u = opt_us[0]
        vect_spline1 = np.array(spline1.f_curve(opt_u + DELTA_U)) - np.array(spline1.f_curve(opt_u))
        # フィレットの直線との接点におけるuが増加する方向のベクトルを算出
        vect_filet1 = np.array(filet.f_curve(filet.u[0] + DELTA_U)) - np.array(filet.f_curve(filet.u[0]))
        
        opt_s = opt_us[1]
        # スプライン2のフィレットとの接点におけるuが増加する方向のベクトルを算出
        vect_spline2 = np.array(spline2.f_curve(opt_s + DELTA_U)) - np.array(spline2.f_curve(opt_s))
        # フィレットのスプラインとの接点におけるuが増加する方向のベクトルを算出
        vect_filet2 = np.array(filet.f_curve(filet.u[-1])) - np.array(filet.f_curve(filet.u[-1] - DELTA_U))
                
        # スプライン1　→　フィレット　→　スプライン2　の順にフィレットの点列は生成される
        # よってスプライン1との接点では、フィレットとスプライン1とのベクトルは反対を向き、スプライン2との接点では同じ方向を向く
        if np.dot(vect_spline1, vect_filet1) < 0:
            # スプライン1とフィレットのベクトルが反対方向なので、スプライン1の向きはもとの方向に増加でOK
            # 始点を接点のu,終点をスプライン1の終点としてトリム
            t_spline1 = trim(spline1, opt_u, spline1.u[-1])   
        else:
            # スプライン1とフィレットのベクトルが同じ方向なので、スプライン1の向きはもとの方向に増加するとNG
            # 始点をスプライン1の始点、終点を接点のuとしてトリム
              t_spline1 = trim(spline1, spline1.u[0], opt_u)
        
        if np.dot(vect_spline2, vect_filet2) >= 0:
            # スプライン2とフィレットのベクトルが同じ方向なので、スプライン2はu増加でOK
            # 始点を接点のs、終点をスプライン終点のsとしてトリム
            t_spline2 = trim(spline2, opt_s, spline2.u[-1])
        else:
            # スプライン2とフィレットのベクトルが逆向きなので、スプライン2のsだとNG
            # スプライン2始点のs、終点を接点のsとしてトリム
            t_spline2 = trim(spline2, spline2.u[0], opt_s)
        
        """
        #for debug
        plt.plot(p1_x, p1_y, "ro")
        plt.plot(p2_x, p2_y, "ko")
        plt.plot(cx, cy, "ko")
        plt.plot(f_x, f_y, "ro")
        plt.plot(fx_est, fy_est, "bo")
        """
              
    except:
        traceback.print_exc()
        pass
    
    return t_spline1, t_spline2, filet


def trim(line, st, ed):

    if (line.line_type == "SLine"):
        x_st = st
        x_ed = ed
        new_x = np.array([x_st, x_ed])
        new_y = line.f_line(new_x)
        
    if (line.line_type == "Spline") or (line.line_type == "Polyline") or \
        (line.line_type == "Arc") or (line.line_type == "Airfoil"):
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
    
    if (line.line_type == "Circle") or (line.line_type == "Ellipse"):
        sita_st = st
        sita_ed = ed

        
    if line.line_type == "Spline":
        return Spline(new_x, new_y)
    
    elif line.line_type == "Polyline":
        return Polyline(new_x, new_y)
    
    elif line.line_type == "SLine":
        return SLine(new_x, new_y)
    
    elif line.line_type == "Arc":
        return Arc(line.r, line.cx, line.cy, sita_st, sita_ed)

    elif line.line_type == "EllipseArc":
        return EllipseArc(line.a, line.b, line.rot, line.cx, line.cy, sita_st-line.rot, sita_ed-line.rot)

    elif line.line_type == "Circle":
        #円をトリムしたオブジェクトは円弧で返す
        return Arc(line.r, line.cx, line.cy, sita_st, sita_ed)

    elif line.line_type == "Ellipse":
        #楕円をトリムしたオブジェクトは楕円弧で返す
        return EllipseArc(line.a, line.b, line.rot, line.cx, line.cy, sita_st-line.rot, sita_ed-line.rot)

    elif line.line_type == "Airfoil":
        #翼形をトリムしたオブジェクトはスプラインを返す
        return Spline(new_x, new_y)

def getTuplePoints(x, y):
    points = []
    i = 0
    while i < len(x):
        points.append((x[i], y[i]))
        i += 1
    return tuple(points)


def getSplineFitPoint(x, y):
    points = []
    i = 0
    while i < len(x):
        points.append((x[i], y[i], 0))
        i += 1
    return tuple(points)    


def exportLine2CommandScript(line):
    temp_str = ""
    
    if (line.line_type == "Spline") or (line.line_type == "Ellipse") or \
        (line.line_type == "EllipseArc") or (line.line_type == "Airfoil"):
        i = 0
        temp_str += "_.SPLINE\n"
        while i < len(line.x_intp):
            temp_str += "%s,%s\n"%(line.x_intp[i], line.y_intp[i])
            i += 1
        temp_str += "\n\n\n"
            
    elif line.line_type == "Polyine":
        i = 0
        temp_str += "_.PLINE\n"
        while i < len(line.x_intp):
            temp_str += "%s,%s\n"%(line.x_intp[i], line.y_intp[i])
            i += 1
        temp_str += "\n\n"
        
    elif line.line_type == "SLine":
        temp_str += "_.LINE\n"
        temp_str += "%s,%s\n"%(line.st[0], line.st[1])
        temp_str += "%s,%s\n"%(line.ed[0], line.ed[1])
        temp_str += "\n"
    
    elif line.line_type == "Arc":
        temp_str += "_.ARC\n"
        temp_str += "C\n"
        temp_str += "%s,%s\n"%(line.cx, line.cy)
        temp_str += "%s,%s\n"%(line.st[0], line.st[1])
        temp_str += "A\n"
        temp_str += "%s\n"%np.degrees((line.sita_ed-line.sita_st))
        
    elif line.line_type == "Circle":
        temp_str += "_.CIRCLE\n"
        temp_str += "%s,%s\n"%(line.cx, line.cy)
        temp_str += "%s\n"%(line.r)
    
    return temp_str 


def exportLine2ModeWorkSpace(msp, layer, line, \
                             color=DXF_COLOR_DEFAULT, \
                             linetypes=DXF_LINETYPES_DEFAULT, \
                             width=DXF_WIDTH_DEFAULT):
    
    attr = {'layer': layer, #レイヤー
            'color': color, #色
            'lineweight':width, #線幅
             'linetype': linetypes #線種
            }
    
    if (line.line_type == "Spline") or (line.line_type == "Ellipse") or line.line_type == "EllipseArc":
        # 楕円、楕円弧はezdxfにはないので、ポリラインで描画する
        if DXF_USE_SPLINE == True:
            # スプラインをスプラインで出力する
            points = getSplineFitPoint(line.x_intp, line.y_intp)
            msp.add_spline(points, dxfattribs = attr)
            
        else:
            # スプラインはサポートされていない場合は、ポリラインとして出力する
            points = getTuplePoints(line.x_intp, line.y_intp)
            msp.add_lwpolyline(points, format="xy", close=False, dxfattribs = attr)

    elif line.line_type == "Polyine":
        points = getTuplePoints(line.x_intp, line.y_intp)
        msp.add_lwpolyline(points, format="xy", close=False, dxfattribs = attr) 
    
    elif line.line_type == "SLine":
        msp.add_line(start=tuple(line.st), end=tuple(line.ed), dxfattribs = attr)
    
    elif line.line_type == "Arc":
        msp.add_arc(center = (line.cx, line.cy), radius = line.r,\
                    start_angle = np.degrees(line.sita_st), end_angle = np.degrees(line.sita_ed), \
                    dxfattribs = attr)
    
    elif line.line_type == "Circle":
        msp.add_circle(center = (line.cx, line.cy), radius = line.r, dxfattribs = attr)

    
def importLinesFromDxf(msp, dxf_object_type):
    line_objs = msp.query(dxf_object_type)
    line_list = []
    
    if not(len(line_objs) == 0):
        if dxf_object_type == 'LINE':
            for line_obj in line_objs:
                temp_line = []
                temp_line.append(line_obj.dxf.start)
                temp_line.append(line_obj.dxf.end)
                temp_line = np.array(temp_line)[:,0:2]
                if norm(temp_line[0,0],temp_line[0,1],temp_line[1,0],temp_line[1,1]) != 0:
                    line = SLine(temp_line[:,0], temp_line[:,1])
                    line_list.append(line)
                        
        elif dxf_object_type == 'SPLINE':
            for line_obj in line_objs:
                control_points = np.array(line_obj.control_points)[:]
                fit_points = np.array(line_obj.fit_points)[:]
                if not(len(control_points) == 0):
                    spline = Spline(control_points[:,0], control_points[:,1]) 
                    line_list.append(spline)
                elif not(len(fit_points) == 0):
                    spline = Spline(fit_points[:,0], fit_points[:,1]) 
                    line_list.append(spline)                    
                
        elif dxf_object_type == 'ARC':
            for line_obj in line_objs:
                r = line_obj.dxf.radius
                cx  = line_obj.dxf.center[0]
                cy  = line_obj.dxf.center[1]
                sita_st = np.radians(line_obj.dxf.start_angle)
                sita_ed = np.radians(line_obj.dxf.end_angle)
                arc = Arc(r, cx, cy, sita_st, sita_ed)
                line_list.append(arc)  

            
        elif dxf_object_type == 'LWPOLYLINE':
                control_points = np.array(line_obj.control_points)[:]
                fit_points = np.array(line_obj.fit_points)[:]
                if not(len(control_points) == 0):
                    polyline = Polyline(control_points[:,0], control_points[:,1]) 
                    line_list.append(polyline)
                elif not(len(fit_points) == 0):
                    polyline = Polyline(fit_points[:,0], fit_points[:,1]) 
                    line_list.append(polyline)     

        return line_list     


def sortLines(line_list, num_st):
    i = 0
    norm_min = np.inf
    line = line_list[num_st]
    used_num = [num_st]
    x0 = line.ed[0]
    y0 = line.ed[1]
    flag_invert = False
    
    sorted_lines = [line]

    while i < len(line_list)-1:
        norm_min = np.inf
        j = 0
        while j < len(line_list):          
            if j in used_num:
                pass
            else:
                line = line_list[j]
                norm_st = norm(x0, y0, line.st[0], line.st[1])
                norm_ed = norm(x0, y0, line.ed[0], line.ed[1])

                if min(norm_st, norm_ed) < norm_min:
                    num = j
                    norm_min = min(norm_st, norm_ed)
                    if norm_st < norm_ed:
                        flag_invert = False
                    else:
                        flag_invert = True
            j += 1
        used_num.append(num)
        line = line_list[num]
        if flag_invert == False:
            x0 = line.ed[0]
            y0 = line.ed[1]
            sorted_lines.append(line)
        else:
            x0 = line.st[0]
            y0 = line.st[1]
            sorted_lines.append(invert(line))
        i += 1
        
    return sorted_lines


def detectCloseLines(line_list, num_st):
    i = 0
    norm_min = np.inf
    line = line_list[num_st]
    used_num = [num_st]
    x0 = line.ed[0]
    y0 = line.ed[1]
    flag_invert = False
    
    lines = [line]
    line_group_list = []

    while i < len(line_list)-1:
        norm_min = np.inf
        j = 0
        while j < len(line_list):          
            if j in used_num:
                pass
            else:
                line = line_list[j]
                norm_st = norm(x0, y0, line.st[0], line.st[1])
                norm_ed = norm(x0, y0, line.ed[0], line.ed[1])

                if min(norm_st, norm_ed) < norm_min:
                    num = j
                    norm_min = min(norm_st, norm_ed)
                    if norm_st < norm_ed:
                        flag_invert = False
                    else:
                        flag_invert = True
            j += 1
        if np.abs(norm_min) > DIST_NEAR:
            line_group_list.append(LineGroup(lines))
            lines = []
        
        line = line_list[num]
        if flag_invert == False:
            x0 = line.ed[0]
            y0 = line.ed[1]
            lines.append(line)
        else:
            x0 = line.st[0]
            y0 = line.st[1]
            lines.append(invert(line))
                
        used_num.append(num)
        i += 1
    
    line_group_list.append(LineGroup(lines))
    
    return line_group_list


def getInclusionList(parent_line, child_line):
    i = 0
    p_list = []
    while i < len(parent_line.x):
        p_list.append((parent_line.x[i], parent_line.y[i]))
        i += 1
    parent_polygon = Polygon(p_list)
    
    inside_x = []
    inside_y = []
    outside_x = []
    outside_y = []
    
    point = Point((child_line.x[0], child_line.y[0]))
    x = [child_line.x[0]]
    y = [child_line.y[0]]
    is_inside = parent_polygon.contains(point)
    
    i = 1
    while i < len(child_line.x):
        point = Point((child_line.x[i], child_line.y[i]))
        if (parent_polygon.contains(point) == True) and (is_inside == True):
            is_inside = True
            x.append(child_line.x[i])
            y.append(child_line.y[i])
        elif (parent_polygon.contains(point) == True) and (is_inside == False):
            is_inside = True
            outside_x.append(x)
            outside_y.append(y)
            x = [child_line.x[i]]
            y = [child_line.y[i]]
        elif (parent_polygon.contains(point) == False) and (is_inside == True):
            is_inside = False
            inside_x.append(x)
            inside_y.append(y)
            x = [child_line.x[i]]
            y = [child_line.y[i]]
        elif (parent_polygon.contains(point) == False) and (is_inside == False):
            is_inside = False
            x.append(child_line.x[i])
            y.append(child_line.y[i])          
        i += 1
    
    if is_inside == True:
        inside_x.append(x)
        inside_y.append(y)
    else:
        outside_x.append(x)
        outside_y.append(y)        
    
    return inside_x, inside_y, outside_x, outside_y
        
 
def checkInclusion(parent_line, child_line):
    inside_x, inside_y, outside_x, outside_y = getInclusionList(parent_line, child_line)
    if len(inside_x) == 0:
        # child_lineはparent_lineの外
        return 0
    elif len(outside_x) == 0:
        # child_lineはparent_lineの内
        return 1
    else:
        # child_lineはparent_lineのと交差
        return 2

# https://qiita.com/wihan23/items/03efd7cd40dfec96a987
def max_min_cross(p1, p2, p3, p4):
    min_ab, max_ab = min(p1, p2), max(p1, p2)
    min_cd, max_cd = min(p3, p4), max(p3, p4)

    if min_ab > max_cd or max_ab < min_cd:
        return False

    return True

# https://qiita.com/wihan23/items/03efd7cd40dfec96a987
def cross_judge(a, b, c, d):
    # x座標による判定
    if not max_min_cross(a[0], b[0], c[0], d[0]):
        return False

    # y座標による判定
    if not max_min_cross(a[1], b[1], c[1], d[1]):
        return False

    tc1 = (a[0] - b[0]) * (c[1] - a[1]) + (a[1] - b[1]) * (a[0] - c[0])
    tc2 = (a[0] - b[0]) * (d[1] - a[1]) + (a[1] - b[1]) * (a[0] - d[0])
    td1 = (c[0] - d[0]) * (a[1] - c[1]) + (c[1] - d[1]) * (c[0] - a[0])
    td2 = (c[0] - d[0]) * (b[1] - c[1]) + (c[1] - d[1]) * (c[0] - b[0])
    return tc1 * tc2 <= 0 and td1 * td2 <= 0


def detectSelfCollision(x, y):
    new_x = [x[0]]
    new_y = [y[0]]
    
    i = 1
    while i < len(x):
        j = i+1
        p1 = [x[i-1], y[i-1]]
        p2 = [x[i], y[i]]
        while j < len(x)-3:
            p3 = [x[j], y[j]]
            p4 = [x[j+1], y[j+1]]
            if cross_judge(p1, p2, p3, p4) == True:
                cx, cy = getCrossPointFromPoint(p1[0], p1[1], p3[0], p3[1], p2[0], p2[1], p4[0], p4[1])
                new_x.append(cx)
                new_y.append(cy)             
                i = j+1
            j += 1
        new_x.append(x[i])
        new_y.append(y[i])
        i += 1
    
    return np.array(new_x), np.array(new_y)

