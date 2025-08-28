# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 13:36:54 2025

@author: hirar
"""

# 外部ライブラリ
import cad_cam_lib as clib
from matplotlib import pyplot as plt
import numpy as np
import ezdxf as ez
from ezdxf import recover
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

# 内部ライブラリ
import airfoilData as af


def example_3_2_1_1(plotGraph):
    #座標点変化量による接線の傾き算出
    m1 = clib.getM1(af.NACA2412_X, af.NACA2412_Y)
    
    #arctan2による接線の傾き算出
    sita1 = clib.getSita1(af.NACA2412_X, af.NACA2412_Y)
    m1_atan2 = np.tan(sita1)
    
    #グラフを描画
    if plotGraph == True:
        fig = plt.figure()
        plt.title("Example of 3.2.1.1")
        ax1 = fig.add_subplot(111)
        ax1.plot(af.NACA2412_X, af.NACA2412_Y, "bo--")
        ax1.quiver(af.NACA2412_X, af.NACA2412_Y, np.cos(sita1), np.sin(sita1),\
                   angles='xy',scale_units='xy',scale=50, width=0.002, color = 'blue')
        ax2 = ax1.twinx()
        ax2.step(af.NACA2412_X, m1, "r")
        ax2.step(af.NACA2412_X, m1_atan2, "k--")
        ax2.set_ylim([-1,1])
        ax2.grid(True)
        ax1.legend(["RawData"],loc="lower right")
        ax2.legend(["Numerical Differentiation", "Arctan2"] ,loc="upper right")
        ax1.set_aspect("equal")
        
    return m1, sita1, m1_atan2 

def example_3_2_2_1(plotGraph):
    #座標点変化量による垂線の傾き算出
    m2 = clib.getM2(af.NACA2412_X, af.NACA2412_Y)
    
    #arctan2による垂線の傾き算出
    sita2 = clib.getSita2(af.NACA2412_X, af.NACA2412_Y)
    m2_atan2 = np.tan(sita2)
    
    #グラフを描画
    if plotGraph == True:
        fig = plt.figure()
        plt.title("Example of 3.2.2.1")
        ax1 = fig.add_subplot(111)
        ax1.plot(af.NACA2412_X, af.NACA2412_Y, "bo--")
        ax1.quiver(af.NACA2412_X, af.NACA2412_Y, np.cos(sita2), np.sin(sita2), \
                   angles='xy',scale_units='xy',scale=50, width=0.002, color = 'blue')
        ax2 = ax1.twinx()
        ax2.step(af.NACA2412_X, m2, "r")
        ax2.step(af.NACA2412_X, m2_atan2, "k--")
        ax2.set_ylim([-500,500])
        ax2.grid(True)
        ax1.legend(["RawData"],loc="lower right")
        ax2.legend(["Numerical Differentiation", "Arctan2"] ,loc="upper right")      
        ax1.set_aspect("equal")
        
    return m2, sita2, m2_atan2     

def example_3_5_1(plotGraph):
    # 直線を作成
    l0 = clib.SLine([-3, 3], [1, 3])
    l1 = clib.SLine([-2, 4], [2, -3])
    
    #　交点を算出
    cx, cy = clib.getCrossPointFromLines(l0.a, l0.b, l1.a, l1.b)
    
    #　グラフを描画
    if plotGraph == True:
        plt.title("Example of 3.5.1")
        plt.plot(l0.x, l0.y, "b-")
        plt.plot(l1.x, l1.y, "b-")
        plt.plot(cx, cy, "go")
        plt.axis("equal")  
    return l0, l1, cx, cy


def example_3_5_2(plotGraph):
    # 直線とスプラインを作成
    line = clib.SLine([0.1, 0.4], [-0.2, 0.3])
    spline = clib.Spline(af.NACA2412_X[0:52], af.NACA2412_Y[0:52])    
    
    #　交点を算出
    u_root, cx, cy = clib.getCrossPointFromCurveLine(line.f_line, spline.f_curve, 0.5)    

    #　グラフを描画
    if plotGraph == True:
        plt.title("Example of 3.5.2")
        plt.plot(line.x, line.y, "b")
        plt.plot(spline.x, spline.y, "b")
        plt.plot(cx, cy, "go")
        plt.axis("equal")  
    return line, spline, cx, cy

def example_3_5_3(plotGraph):
    # 交差するスプラインを2つ作成
    spline1 = clib.Spline(af.NACA2412_X[0:52], af.NACA2412_Y[0:52] - 0.03)
    spline2 = clib.Spline(af.NACA2412_X[53:-1], af.NACA2412_Y[53:-1] + 0.03) 
    
    #　交点を算出
    u_root1, s_root1, cx1, cy1 = clib.getCrossPointFromCurves(spline1.f_curve, spline2.f_curve, 0.7, 0.1)
    u_root2, s_root2, cx2, cy2 = clib.getCrossPointFromCurves(spline1.f_curve, spline2.f_curve, 0.1, 0.7)

    #　グラフを描画
    if plotGraph == True:
        plt.title("Example of 3.5.2")
        plt.plot(spline1.x, spline1.y, "b")
        plt.plot(spline2.x, spline2.y, "b")
        plt.plot(cx1, cy1, "go")
        plt.plot(cx2, cy2, "ro")
        plt.axis("equal")  
    return  spline1, spline2, u_root1, s_root1


def example_3_6_1_1(plotGraph):
    #スプライン関数/ポリライン関数を生成
    spline = clib.Spline(af.NACA2412_X, af.NACA2412_Y)
    polyline = clib.Polyline(af.NACA2412_X, af.NACA2412_Y)

    #補完点を5000点生成。
    interp_point = np.linspace(0,1,5000)
    x_s, y_s = spline.getPoint(interp_point)
    x_p, y_p = polyline.getPoint(interp_point)
 
    #グラフを描画
    if plotGraph == True:
        plt.title("Example of 3.6.1.1")
        plt.plot(x_s, y_s, "bo-")
        plt.plot(x_p, y_p, "ko-")
        plt.plot(af.NACA2412_X, af.NACA2412_Y, "ro")
        plt.legend(["spline", "polyline", "RawData"])
        plt.axis("equal")    
    
    return x_s, y_s, x_p, y_p, interp_point


def example_3_6_1_2(plotGraph):
    #スプライン関数/ポリライン関数を生成
    spline = clib.Spline(af.NACA2412_X, af.NACA2412_Y)
    polyline = clib.Polyline(af.NACA2412_X, af.NACA2412_Y)

    #補完点を5000点生成。
    interp_point = np.linspace(0,1,5000)
    x_s, y_s = spline.getPoint(interp_point)
    x_p, y_p = polyline.getPoint(interp_point)
    
    #傾きの補完点を生成。
    x_s, m1_s = spline.getM1(interp_point)
    x_p, m1_p = polyline.getM1(interp_point)
    
    #グラフを描画
    if plotGraph == True:
        sita_s = np.arctan(m1_s)
        
        fig = plt.figure()
        plt.title("Example of 3.6.1.2")
        ax1 = fig.add_subplot(111)
        ax1.plot(x_s, y_s, "bo-")
        ax1.plot(x_p, y_p, "ko-")

        ax2 = ax1.twinx()
        ax2.plot(x_s, m1_s, "b--")
        ax2.step(x_p, m1_p, "k--")
        ax2.set_ylim([-1,1])
        ax2.grid(True)
        ax1.plot(af.NACA2412_X, af.NACA2412_Y, "ro")
        ax1.quiver(x_s, y_s, np.cos(sita_s), np.sin(sita_s), \
           angles='xy',scale_units='xy',scale=3000, width=0.002, color = 'blue')
        plt.legend(["spline", "polyline", "RawData"])
        ax1.set_aspect("equal")    
    
    return m1_s, m1_p

def example_3_6_1_3(plotGraph):
    #スプライン関数/ポリライン関数を生成
    spline = clib.Spline(af.NACA2412_X, af.NACA2412_Y)
    polyline = clib.Polyline(af.NACA2412_X, af.NACA2412_Y)

    #補完点を5000点生成。
    interp_point = np.linspace(0,1,5000)
    x_s, y_s = spline.getPoint(interp_point)
    x_p, y_p = polyline.getPoint(interp_point)
    
    #垂線の傾きの補完点を生成。
    x_s, m2_s = spline.getM2(interp_point)
    x_p, m2_p = polyline.getM2(interp_point)
    
    #グラフを描画
    if plotGraph == True:
        sita_s = np.arctan(m2_s)
        
        fig = plt.figure()
        plt.title("Example of 3.6.1.3")
        ax1 = fig.add_subplot(111)
        ax1.plot(x_s, y_s, "bo-")
        ax1.plot(x_p, y_p, "ko-")
        ax1.quiver(x_s, y_s, np.cos(sita_s), np.sin(sita_s), \
           angles='xy',scale_units='xy',scale=2000, width=0.002, color = 'blue')
        ax2 = ax1.twinx()
        ax2.plot(x_s, m2_s, "b--")
        ax2.step(x_p, m2_p, "k--")
        ax2.set_ylim([-500,500])
        ax2.grid(True)
        ax1.plot(af.NACA2412_X, af.NACA2412_Y, "ro")
        plt.legend(["spline", "polyline", "RawData"])
        ax1.set_aspect("equal")    
    
    return m2_s, m2_p


def example_3_7(plotGraph):
    # 直線とスプラインと楕円を作成
    line = clib.SLine([-0.5, -0.1], [-0.2, 0.3])
    spline = clib.Spline(af.NACA2412_X, af.NACA2412_Y)
    ellipse = clib.Ellipse(0.5, 0.25, 0, 1, 1)
    
    # 点列の向きを反転
    i_line = clib.invert(line)
    i_spline = clib.invert(spline)
    i_ellipse = clib.invert(ellipse)
    
    # グラフを描画
    if plotGraph == True:    
        plt.title("Example of 3.7")
        plt.plot(line.x, line.y, "b")
        plt.plot(spline.x, spline.y, "b")
        plt.plot(ellipse.x, ellipse.y, "b")

        plt.plot(i_line.x, i_line.y, "r--")
        plt.plot(i_spline.x, i_spline.y, "r--")
        plt.plot(i_ellipse.x, i_ellipse.y, "r--")
        
        plt.quiver(line.x[0], line.y[0], line.x[1]-line.x[0], line.y[1]-line.y[0], \
           angles='xy',scale_units='xy',scale=3, width=0.01, color = 'blue')
        plt.quiver(spline.x[0], spline.y[0], spline.x[1]-spline.x[0], spline.y[1]-spline.y[0], \
           angles='xy',scale_units='xy',scale=0.15, width=0.01, color = 'blue')
        plt.quiver(ellipse.x[0], ellipse.y[0], ellipse.x[1]-ellipse.x[0], ellipse.y[1]-ellipse.y[0], \
           angles='xy',scale_units='xy',scale=0.1, width=0.01, color = 'blue')

        plt.quiver(i_line.x[0], i_line.y[0], i_line.x[1]-i_line.x[0], i_line.y[1]-i_line.y[0], \
           angles='xy',scale_units='xy',scale=3, width=0.01, color = 'red')
        plt.quiver(i_spline.x[0], i_spline.y[0], i_spline.x[1]-i_spline.x[0], i_spline.y[1]-i_spline.y[0], \
           angles='xy',scale_units='xy',scale=0.15, width=0.01, color = 'red')
        plt.quiver(i_ellipse.x[0], i_ellipse.y[0], i_ellipse.x[1]-i_ellipse.x[0], i_ellipse.y[1]-i_ellipse.y[0], \
           angles='xy',scale_units='xy',scale=0.1, width=0.01, color = 'red')
        
        
        plt.axis("equal")
        plt.show()        

def example_4_1(plotGraph):
    # ファイルから座標点列を読み込み
    line = clib.importFromText("NACA2412.csv", "spline")
    
    # グラフを描画
    if plotGraph == True:
        plt.title("Example of 4.1")
        plt.plot(line.x, line.y)
        plt.axis("equal")
        plt.show()
    
    return line

def example_4_2(plotGraph):
    # 直線とスプラインと楕円を作成
    line = clib.SLine([0.1, 0.4], [-0.2, 0.3])
    spline = clib.Spline(af.NACA2412_X, af.NACA2412_Y)
    ellipse = clib.Ellipse(0.5, 0.25, 0, 1, 1)
    
    # x,y方向に0.1移動する
    mx = 0.1
    my = 0.1
    
    # 移動後の直線とスプラインを作成
    m_line = clib.move(line, mx, my)
    m_spline = clib.move(spline, mx, my)
    m_ellipse = clib.move(ellipse, mx,my)
    
    # グラフを描画
    if plotGraph == True:
        plt.title("Example of 4.2")
        plt.plot(line.x, line.y, "b--")
        plt.plot(spline.x, spline.y, "b--")
        plt.plot(ellipse.x, ellipse.y, "b--")
        plt.plot(m_line.x, m_line.y, "r")
        plt.plot(m_spline.x, m_spline.y, "r")
        plt.plot(m_ellipse.x, m_ellipse.y, "r")
        plt.axis("equal")
        plt.show()

def example_4_3(plotGraph):
    # 直線とスプラインと楕円を作成
    line = clib.SLine([0.1, 0.4], [-0.2, 0.3])
    spline = clib.Spline(af.NACA2412_X, af.NACA2412_Y)
    ellipse = clib.Ellipse(0.5, 0.25, 0, 1, 1)
    
    # (x,y) = (0.5, 0.5)を中心に時計回りに45度回転する
    rx = 0.5
    ry = 0.5
    sita = np.radians(45)
    
    # 移動後の直線とスプラインを作成
    m_line = clib.rotate(line, sita, rx, ry)
    m_spline = clib.rotate(spline, sita, rx, ry)
    m_ellipse = clib.rotate(ellipse, sita, rx, ry)
    
    # グラフを描画
    if plotGraph == True:
        plt.title("Example of 4.3")
        plt.plot(line.x, line.y, "b--")
        plt.plot(spline.x, spline.y, "b--")
        plt.plot(ellipse.x, ellipse.y, "b--")
        plt.plot(m_line.x, m_line.y, "r")
        plt.plot(m_spline.x, m_spline.y, "r")
        plt.plot(m_ellipse.x, m_ellipse.y, "r")
        plt.plot(rx, ry, "ro")
        plt.axis("equal")
        plt.show()    

def example_4_4(plotGraph):
    # 直線を作成
    l0 = clib.SLine([-3, 3], [-3, 3])
    l1 = clib.SLine([-3, 3], [3, -3])
    
    print(l0.a, l0.b, np.degrees(l0.sita))
    print(l1.a, l1.b, np.degrees(l1.sita))
    
    #　グラフを描画
    if plotGraph == True:
        plt.title("Example of 4.4")
        plt.plot(l0.x, l0.y, "b")
        plt.plot(l1.x, l1.y, "r")
        plt.axis("equal")  
        plt.grid(True)
    return l0, l1


def example_4_6(plotGraph):
    #半径2, 中心(1,1)とした円を作成
    r = 2
    rx = 1
    ry = 1
    circle = clib.Circle(r, rx, ry)
    
    #補完点を200点生成。
    interp_point = np.linspace(0,1,200)  
    x_i, y_i = circle.getPoint(interp_point)

    #　グラフを描画
    if plotGraph == True:
        plt.title("Example of 4.6")
        plt.plot(circle.x, circle.y, "bo")
        plt.plot(x_i, y_i, "r--")
        plt.axis("equal")  
        plt.grid(True)

def example_4_7(plotGraph):
    #長軸2, 短軸1, 中心(1,1)とし、時計回りに30度傾けた楕円を作成
    a = 2
    b = 1
    rx = 1
    ry = 1
    rot = np.radians(30)
    ellipse = clib.Ellipse(a, b, rot, rx, ry)
    
    #補完点を200点生成。
    interp_point = np.linspace(0,1,200)  
    x_i, y_i = ellipse.getPoint(interp_point)

    #　グラフを描画
    if plotGraph == True:
        plt.title("Example of 4.7")
        plt.plot(ellipse.x, ellipse.y, "bo")
        plt.plot(x_i, y_i, "r--")
        plt.axis("equal")  
        plt.grid(True)


def example_4_8(plotGraph):
    # 直線とスプラインを作成
    line = clib.SLine([0.1, 0.4], [-0.2, 0.3])
    spline = clib.Spline(af.NACA2412_X, af.NACA2412_Y)
    
    # 0.01だけオフセットする
    d = 0.01
    
    # オフセット後の直線とスプラインを作成
    o_line = clib.offset(line, d)
    o_spline = clib.offset(spline, -d)
    
    # グラフを描画
    if plotGraph == True:
        plt.title("Example of 4.8")
        plt.plot(line.x, line.y, "b--")
        plt.plot(spline.x, spline.y, "b--")
        plt.plot(o_line.x, o_line.y, "r")
        plt.plot(o_spline.x, o_spline.y, "r")
        plt.axis("equal")
        plt.show()    


def example_4_9_2(plotGraph):
    l0 = clib.SLine([-3,-1], [5,5])
    l1 = clib.SLine([-1,-2], [5.5,6])
    
    new_l0, new_l1, filet = clib.filetLines(l0, l1, 0.2)
    
    #グラフを描画
    if plotGraph == True:    
        plt.plot(l0.x, l0.y, "b")
        plt.plot(l1.x, l1.y, "b")
        plt.plot(filet.cx, filet.cy, "go")
        plt.plot(filet.x, filet.y, "ro-")
        plt.plot(new_l0.x, new_l0.y, "ro--")
        plt.plot(new_l1.x, new_l1.y, "ro--")
        plt.axis("equal")


def example_4_9_3(plotGraph):
    # 直線とスプラインを作成
    line = clib.SLine([-0.0, 0.2], [-0.4, 0.4])
    spline = clib.Spline(af.NACA2412_X[0:52], af.NACA2412_Y[0:52])

    # 直線とスプラインの交点を、r=0.02でフィレットする。
    # フィレット箇所は、直線に対して第一象限とする
    # 複数交点を持つ場合に備え、探索始点はスプラインの40%位置とする
    t_line, t_spline, filet = clib.filetLineCurve(line, spline, 0.02, 1, 0.4)
    
    #グラフを描画
    if plotGraph == True:    
        plt.plot(line.x, line.y, "b")
        plt.plot(spline.x, spline.y, "b")
        plt.plot(filet.x, filet.y, "r--")
        plt.plot(t_spline.x, t_spline.y, "r--")
        plt.plot(t_line.x, t_line.y, "r--")
        plt.axis("equal")
    
    
def example_4_9_4(plotGraph):
    # 交差するスプラインを2つ作成
    spline1 = clib.Spline(af.NACA2412_X[0:52], af.NACA2412_Y[0:52] - 0.03)
    spline2 = clib.Spline(af.NACA2412_X[51:-1], af.NACA2412_Y[51:-1] + 0.03)

    # スプラインどうしの交点を、r=0.005でフィレットする。
    # フィレット箇所は、スプライン1に対して第一象限とする
    # 複数交点を持つ場合に備え、探索始点はスプライン1、スプライン2の10%位置とする    
    t_spline1, t_spline2, filet = clib.filetCurves(spline1, spline2, 0.005, 1, 0.1, 0.1)
    
    #グラフを描画
    if plotGraph == True:    
        plt.plot(spline1.x, spline1.y, "b")
        plt.plot(spline2.x, spline2.y, "b")
        plt.plot(filet.x, filet.y, "r--")
        plt.plot(t_spline1.x, t_spline1.y, "r--")
        plt.plot(t_spline2.x, t_spline2.y, "r--")        
        plt.axis("equal")
    
  
def example_4_10(plotGraph):
    # 直線とスプラインを作成
    line = clib.SLine([0.1, 0.4], [-0.2, 0.3])
    spline = clib.Spline(af.NACA2412_X, af.NACA2412_Y)
    circle = clib.Circle(0.2, 0.5, 0.5)
    
    # (0.5, 0.5）を中心に1.5倍した図形を作成
    s = 1.5
    x0 = 0.5
    y0 = 0.5
    
    # 移動後の直線とスプラインを作成
    s_line = clib.scale(line, s, x0, y0)
    s_spline = clib.scale(spline, s, x0, y0)
    s_circle = clib.scale(circle, s, x0, y0)
    
    # グラフを描画
    if plotGraph == True:
        plt.title("Example of 4.10")
        plt.plot(line.x, line.y, "b--")
        plt.plot(spline.x, spline.y, "b--")
        plt.plot(circle.x, circle.y, "b--")
        plt.plot(s_line.x, s_line.y, "r")
        plt.plot(s_spline.x, s_spline.y, "r")
        plt.plot(s_circle.x, s_circle.y, "r")
        plt.axis("equal")
        plt.show()        

def example_4_11_1(plotGraph):
    # 線分を作成
    line = clib.SLine([-3, 2], [1, 5])

    # x:-1~0.5の範囲でトリム
    x_st = -1
    x_ed = 0.5
    t_line = clib.trim(line, x_st, x_ed)
    
    # グラフを描画
    if plotGraph == True:
        plt.title("Example of 4.11.1")
        plt.plot(line.x, line.y, "b--")
        plt.plot(t_line.x, t_line.y, "r")
        plt.axis("equal")
        plt.show()         

def example_4_11_2(plotGraph):
    # 交差するスプラインを2つ作成
    spline1 = clib.Spline(af.NACA2412_X[0:52], af.NACA2412_Y[0:52] - 0.03)
    spline2 = clib.Spline(af.NACA2412_X[53:-1], af.NACA2412_Y[53:-1] + 0.03) 
    
    #　交点を算出
    u_root1, s_root1, cx1, cy1 = clib.getCrossPointFromCurves(spline1.f_curve, spline2.f_curve, 0.7, 0.1)
    u_root2, s_root2, cx2, cy2 = clib.getCrossPointFromCurves(spline1.f_curve, spline2.f_curve, 0.1, 0.7)

    # 交点でトリムしたスプラインを作成
    u_st = 0
    u_ed = u_root1
    s_st = s_root1
    s_ed = 1
    t_spline1 = clib.trim(spline1, u_st, u_ed)
    t_spline2 = clib.trim(spline2, s_st, s_ed)
    
    # グラフを描画
    if plotGraph == True:
        plt.title("Example of 4.11.2")
        plt.plot(spline1.x, spline1.y, "b--")
        plt.plot(spline2.x, spline2.y, "b--")
        plt.plot(t_spline1.x, t_spline1.y, "r")
        plt.plot(t_spline2.x, t_spline2.y, "r")
        plt.axis("equal")
        plt.show()          


def example_4_11_3(plotGraph):
    #半径2, 中心(1,1)の円を作成
    r = 2
    rx = 1
    ry = 1
    circle = clib.Circle(r, rx, ry)

    #180度～405度の範囲をトリム
    sita_st = np.radians(180)
    sita_ed = np.radians(405)    
    t_arc = clib.trim(circle, sita_st, sita_ed)
    
    # グラフを描画
    if plotGraph == True:
        plt.title("Example of 4.11.4")
        plt.plot(circle.x, circle.y, "b--")
        plt.plot(t_arc.x, t_arc.y, "r")
        plt.axis("equal")
        plt.show()       


def example_4_11_4(plotGraph):
    #長軸2, 短軸1, 中心(1,1)とし、時計回りに30度傾けた楕円を作成
    a = 2
    b = 1
    rx = 1
    ry = 1
    rot = np.radians(30)
    ellipse = clib.Ellipse(a, b, rot, rx, ry)

    #180度～270度の範囲をトリム
    sita_st = np.radians(180)
    sita_ed = np.radians(270)    
    t_spline = clib.trim(ellipse, sita_st, sita_ed)
    
    # グラフを描画
    if plotGraph == True:
        plt.title("Example of 4.11.4")
        plt.plot(ellipse.x, ellipse.y, "b--")
        plt.plot(t_spline.x, t_spline.y, "r")
        plt.axis("equal")
        plt.show()       
        

def example_4_12_2(plotGraph):
    # ezdxfのモデルワークスペースオブジェクトを生成
    doc = ez.new('R2010', setup=True)
    msp = doc.modelspace()
    
    # レイヤーを追加
    attr = {'color': 0, #色
            'lineweight':2, #線幅
            'linetype': 'Continuous' #線種
    }
    
    doc.layers.new(name="layer0",  dxfattribs = attr) # レイヤーを追加
    
    # 直線とスプラインと楕円を作成
    line = clib.SLine([0.1, 0.4], [-0.2, 0.3])
    spline = clib.Spline(af.NACA2412_X, af.NACA2412_Y)
    ellipse = clib.Ellipse(0.5, 0.25, 0, 1, 1)
    
    # スプラインに補完点を追加
    u_intp = np.linspace(0, 1, 1000) 
    spline.setIntporatePoints(u_intp)
    
    # モデルワークスペースに作成した直線とスプラインと楕円を追加
    clib.exportLine2ModeWorkSpace(msp, "layer0", line)
    clib.exportLine2ModeWorkSpace(msp, "layer0", spline)
    clib.exportLine2ModeWorkSpace(msp, "layer0", ellipse)
    
    # dxfを出力
    doc.saveas('example_4_12_2.dxf')
    
    # dxfファイルをCADを使わずにグラフで確認
    if plotGraph == True:
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ctx = RenderContext(doc)
        out = MatplotlibBackend(ax)
        Frontend(ctx, out).draw_layout(msp, finalize=True)
        fig.show()        


if __name__ == '__main__':
    #example_3_2_1_1(True)
    #example_3_2_2_1(True)
    #example_3_5_1(True)
    #example_3_5_2(True)
    #example_3_5_3(True)
    #example_3_6_1_1(True)
    #example_3_6_1_2(True)
    #example_3_6_1_3(True)
    #example_3_7(True)
    #example_4_1(True)
    #example_4_2(True)
    #example_4_3(True)
    #example_4_4(True)
    #example_4_6(True)
    #example_4_7(True)
    #example_4_8(True)
    #example_4_9_2(True)
    #example_4_9_3(True)
    #example_4_9_4(True)
    #example_4_10(True)
    #example_4_11_1(True)
    #example_4_11_2(True)
    #example_4_11_3(True)
    #example_4_11_4(True)
    example_4_12_2(True)
    