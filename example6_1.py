# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 22:45:45 2025

@author: hirar
"""

# 外部ライブラリ
from matplotlib import pyplot as plt
import numpy as np
import ezdxf as ez
from ezdxf import recover
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import copy

# 内部ライブラリ
import cad_cam_lib as clib

def mekaLib(plotGraph):
    ration = 0.3 # 翼型混合比
    chord = 600 # コード長 [mm]
    t_plank = 2 # プランク厚 [mm]
    
    x_plank_u = chord*0.7 # 上面プランク端 [mm]
    x_plank_l = chord*0.2 # 下面プランク端 [mm]
    
    st_width = 3 # ストリンガー幅 [mm]
    st_hight = 4 # ストリンガー高 [mm]
    x_st1 = x_plank_u-20 #[mm]
    x_st2 = x_plank_l-20 #[mm]
    
    cx = chord*0.35 # 桁穴位置 [mm]
    cr = 25 # 桁穴径（半径）
    
    # DAE51, NACA2412、およびそれらをXFLRで20%混合した座標点を読み込み
    x1, y1 = clib.importFromText("foils/dae51.dat", '\t', 1)
    x2, y2 = clib.importFromText("foils/naca2412.dat", ' ', 1)
    
    # DAE51とNACA2412の翼形クラスのインスタンスを作成
    dae51 = clib.Airfoil(x1, y1)
    naca2412 = clib.Airfoil(x2, y2)
    
    # 翼型混合&もととなるスプライン作成
    airfoil = clib.mixAirfoilXFLR(dae51, naca2412, ration)
    outer = clib.Spline(airfoil.x_intp, airfoil.y_intp) # datファイルは座標点が荒いので補完座標で作成
    upper = clib.Spline(airfoil.ux, airfoil.uy)
    lower = clib.Spline(airfoil.lx, airfoil.ly)
    center = clib.Spline(airfoil.cx, airfoil.cy)
    
    # コード倍に拡大
    outer = clib.scale(outer, chord, 0, 0)
    upper = clib.scale(upper, chord, 0, 0)
    lower = clib.scale(lower, chord, 0, 0)
    center = clib.scale(center, chord, 0, 0)
    
    # プランクを生成
    plank = clib.offset(outer, t_plank)
    
    y_plank_u = max(plank.getYfromX(x_plank_u))
    y_plank_l = min(plank.getYfromX(x_plank_l))
    y_outer_u = max(outer.getYfromX(x_plank_u))
    y_outer_l = min(outer.getYfromX(x_plank_l))
    
    u_plank_u = min(plank.getUfromX(x_plank_u))
    u_plank_l = max(plank.getUfromX(x_plank_l))
    plank = clib.trim(plank, u_plank_u, u_plank_l)
    
    line_plank_u = clib.SLine([x_plank_u, x_plank_u], [y_plank_u, y_outer_u])
    line_plank_l = clib.SLine([x_plank_l, x_plank_l], [y_plank_l, y_outer_l])
    
    # ストリンガーを作成
    # 今回はデモなので、Y軸に並行にストリンガーを作成する
    y_st1_st = max(plank.getYfromX(x_st1))
    y_st2_st = min(plank.getYfromX(x_st2))
    y_st1_ed = max(plank.getYfromX(x_st1 + st_width))
    y_st2_ed = min(plank.getYfromX(x_st2 + st_width))
    st1 = clib.Polyline([x_st1, x_st1, x_st1 + st_width, x_st1 + st_width],\
                        [y_st1_st, y_st1_st-st_hight, y_st1_ed-st_hight, y_st1_ed])
    st2 = clib.Polyline([x_st2, x_st2, x_st2 + st_width, x_st2 + st_width],\
                        [y_st2_st, y_st2_st+st_hight, y_st2_ed+st_hight, y_st2_ed])
    
    # 桁穴作成
    cy = center.getYfromX(cx)
    circle = clib.Circle(cr, cx, cy)
    
    
    plt.plot(outer.x, outer.y)
    plt.plot(upper.x, upper.y)
    plt.plot(lower.x, lower.y)
    plt.plot(center.x, center.y)
    plt.plot(plank.x, plank.y)
    plt.plot(line_plank_u.x, line_plank_u.y)
    plt.plot(line_plank_l.x, line_plank_l.y)
    plt.plot(line_plank_u.x, line_plank_u.y)
    plt.plot(st1.x, st1.y)
    plt.plot(st2.x, st2.y)
    plt.plot(circle.x, circle.y)
    plt.axis("equal")
