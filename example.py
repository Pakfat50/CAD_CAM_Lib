# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 13:36:54 2025

@author: hirar
"""

# 外部ライブラリ
import cad_cam_lib as clib
from matplotlib import pyplot as plt
import numpy as np

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


def example_3_6_1_1(plotGraph):
    #スプライン関数/ポリライン関数のデータを計算。
    tck_3d, u_3d = clib.getSplineData(af.NACA2412_X, af.NACA2412_Y, 3) #スプライン関数
    tck_1d, u_1d = clib.getSplineData(af.NACA2412_X, af.NACA2412_Y, 1) #ポリライン関数

    #補完関数を生成。
    spline_func = clib.getInterpFunc(tck_3d, 0)
    polyline_func = clib.getInterpFunc(tck_1d, 0)    

    #補完点を5000点生成。
    interp_point = np.linspace(0,1,5000)
    spline_point = spline_func(interp_point)
    polyline_point = polyline_func(interp_point)
    
    #スプライン補完した座標点列
    x_s = spline_point[0]
    y_s = spline_point[1]
    
    #ポリライン補完した座標点列
    x_p = polyline_point[0]
    y_p = polyline_point[1]
 
    #グラフを描画
    if plotGraph == True:
        fig = plt.figure()
        plt.title("Example of 3.6.1.1")
        ax1 = fig.add_subplot(111)
        ax1.plot(x_s, y_s, "bo-")
        ax1.plot(x_p, y_p, "ko-")
        ax1.plot(af.NACA2412_X, af.NACA2412_Y, "ro")
        plt.legend(["spline", "polyline", "RawData"])
        ax1.set_aspect("equal")    
    
    return x_s, y_s, x_p, y_p, interp_point, tck_3d, tck_1d


def example_3_6_1_2(plotGraph):
    #補完処理(3.6.1.1項)
    x_s, y_s, x_p, y_p, interp_point, tck_3d, tck_1d = example_3_6_1_1(False)

    #スプライン関数/ポリライン関数の導関数を算出
    diff_spline_func = clib.getInterpFunc(tck_3d, 1)
    diff_polyline_func = clib.getInterpFunc(tck_1d, 1)
    
    #媒介変数uによるx,yの微分値を計算
    diff_s = diff_spline_func(interp_point)
    diff_p = diff_polyline_func(interp_point)
    
    #スプライン補完関数の微分からdy/dxを計算
    dx_du_s = diff_s[0]
    dy_du_s = diff_s[1]
    m1_s = dy_du_s/dx_du_s
    
    #ポリライン補完関数の微分からdy/dxを計算
    dx_du_p = diff_p[0]
    dy_du_p = diff_p[1]
    m1_p = dy_du_p/dx_du_p    
    
    #グラフを描画
    if plotGraph == True:
        sita_s = np.arctan(m1_s)
        
        fig = plt.figure()
        plt.title("Example of 3.6.1.2")
        ax1 = fig.add_subplot(111)
        ax1.plot(x_s, y_s, "bo-")
        ax1.plot(x_p, y_p, "ko-")
        ax1.quiver(x_s, y_s, np.cos(sita_s), np.sin(sita_s), \
           angles='xy',scale_units='xy',scale=8000, width=0.002, color = 'blue')
        ax2 = ax1.twinx()
        ax2.plot(x_s, m1_s, "b--")
        ax2.step(x_p, m1_p, "k--")
        ax2.set_ylim([-1,1])
        ax2.grid(True)
        ax1.plot(af.NACA2412_X, af.NACA2412_Y, "ro")
        plt.legend(["spline", "polyline", "RawData"])
        ax1.set_aspect("equal")    
    
    return m1_s, m1_p

def example_3_6_1_3(plotGraph):
    #補完処理(3.6.1.1項)
    x_s, y_s, x_p, y_p, interp_point, tck_3d, tck_1d = example_3_6_1_1(False)

    #スプライン関数/ポリライン関数の導関数を算出
    diff_spline_func = clib.getInterpFunc(tck_3d, 1)
    diff_polyline_func = clib.getInterpFunc(tck_1d, 1)
    
    #媒介変数uによるx,yの微分値を計算
    diff_s = diff_spline_func(interp_point)
    diff_p = diff_polyline_func(interp_point)
    
    #スプライン補完関数の微分から垂線の傾きm2を計算
    dx_du_s = diff_s[0]
    dy_du_s = diff_s[1]
    m2_s = -dx_du_s/dy_du_s
    
    #ポリライン補完関数の微分から垂線の傾きm2を計算
    dx_du_p = diff_p[0]
    dy_du_p = diff_p[1]
    m2_p = -dx_du_p/dy_du_p    
    
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

def example_4_1(plotGraph):
    line = clib.importFromText("NACA2412.csv", "spline")
    
    if plotGraph == True:
        plt.title("Example of 4.1")
        plt.plot(line.x, line.y)
        plt.axis("equal")
        plt.show()
    
    return line


def example_4_9_2(plotGraph):
    l0_x = [-3,-1]
    l0_y = [5,5]
    l1_x = [-1,-2]
    l1_y = [5.5,6]
    
    x_intp, y_intp, new_l0_x, new_l0_y, new_l1_x, new_l1_y, f_x, f_y = clib.filetLines(l0_x, l0_y, l1_x, l1_y, 0.2)
    
    #グラフを描画
    if plotGraph == True:    
        plt.plot(l0_x, l0_y, "b")
        plt.plot(l1_x, l1_y, "b")
        plt.plot(f_x, f_y, "go")
        plt.plot(x_intp, y_intp, "ro-")
        plt.plot(new_l0_x, new_l0_y, "ro--")
        plt.plot(new_l1_x, new_l1_y, "ro--")
        plt.axis("equal")


def example_4_9_3(plotGraph):
    l_x = [0.5, 0.8]
    l_y = [-0.3, 0.6]
    
    #スプライン関数/ポリライン関数のデータを計算。
    tck, u = clib.getSplineData(af.NACA2412_X, af.NACA2412_Y, 3) #スプライン関数

    #補完関数を生成。
    a = (l_y[1] - l_y[0])/(l_x[1] - l_x[0])
    b = -a*l_x[0] + l_y[0]
    
    opt_xu, p1_x, p1_y, p2_x, p2_y, cx, cy, fx, fy, x_intp, y_intp = clib.filetLineCurve(a, b, tck, 0.3, 3, 0.05)
    print(opt_xu)
    
    
    #グラフを描画
    if plotGraph == True:    
        plt.plot(l_x, l_y, "b")
        plt.plot(af.NACA2412_X, af.NACA2412_Y, "g")
        plt.plot(p1_x, p1_y, "ro")
        plt.plot(p2_x, p2_y, "ko")
        plt.plot(cx, cy, "ko")
        plt.plot(fx, fy, "ro")
        plt.plot(x_intp, y_intp)
        #plt.plot(x_intp, y_intp, "ro-")
        #plt.plot(new_l0_x, new_l0_y, "ro--")
        #plt.plot(new_l1_x, new_l1_y, "ro--")
        plt.axis("equal")
    

if __name__ == '__main__':
    
    #example_3_2_1_1(True)
    #example_3_2_2_1(True)
    #example_3_6_1_1(True)
    #example_3_6_1_2(True)
    #example_3_6_1_3(True)
    #example_3_6_1_1(True)
    #example_4_1(True)
    #example_4_9_2(True)
    example_4_9_3(True)