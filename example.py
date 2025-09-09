# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 13:36:54 2025

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
import airfoilData as af
import cad_cam_lib as clib


def meka_rib_demo(plotGraph):
    ############################ リブ情報 ###################################
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
    
    tr_dist = 10 # トラス肉抜き幅 [mm]
    tr_angle = np.radians(60) # トラス角度 [rad]
    fr = 4 # フィレット半径
    area_min = 500 # 肉抜き最小面積 [mm^2]
    
    rot = np.radians(3) # 取付迎角 [rad]
    
    ############################ 翼型生成 #################################
    # DAE51, NACA2412の座標点を読み込み
    x1, y1 = clib.importFromText("foils/dae51.dat", '\t', 1)
    x2, y2 = clib.importFromText("foils/naca2412.dat", ' ', 1)
    
    # DAE51とNACA2412の翼形クラスのインスタンスを作成
    dae51 = clib.Airfoil(x1, y1)
    naca2412 = clib.Airfoil(x2, y2)
    
    ############################ 外形生成 #################################
    # 翼型混合
    airfoil = clib.mixAirfoilXFLR(dae51, naca2412, ration)
    # もとになるスプラインを生成
    outer = clib.Spline(airfoil.x_intp, airfoil.y_intp) # datファイルは座標点が荒いので補完座標で作成
    upper = clib.Spline(airfoil.ux, airfoil.uy)
    lower = clib.Spline(airfoil.lx, airfoil.ly)
    center = clib.Spline(airfoil.cx, airfoil.cy)
    
    # コード倍に拡大
    outer = clib.scale(outer, chord, 0, 0)
    upper = clib.scale(upper, chord, 0, 0)
    lower = clib.scale(lower, chord, 0, 0)
    center = clib.scale(center, chord, 0, 0)
    
    ############################ プランク生成 ###############################
    plank_l = clib.offset(outer, t_plank)
    
    # プランク端面を生成
    y_plank_u = max(plank_l.getYfromX(x_plank_u))
    y_plank_l = min(plank_l.getYfromX(x_plank_l))
    y_outer_u = max(outer.getYfromX(x_plank_u))
    y_outer_l = min(outer.getYfromX(x_plank_l))
    line_plank_u = clib.SLine([x_plank_u, x_plank_u], [y_plank_u, y_outer_u])
    line_plank_l = clib.SLine([x_plank_l, x_plank_l], [y_plank_l, y_outer_l])
    
    # プランク内側を生成
    u_plank_u = min(plank_l.getUfromX(x_plank_u))
    u_plank_l = max(plank_l.getUfromX(x_plank_l))
    plank_l = clib.trim(plank_l, u_plank_u, u_plank_l)
    
    # プランク外側を生成
    u_outer_u = min(outer.getUfromX(x_plank_u))
    u_outer_l = max(outer.getUfromX(x_plank_l))
    plank_u = clib.trim(outer, u_outer_u, u_outer_l)
    
    plank = clib.LineGroup([plank_u, plank_l, line_plank_u, line_plank_l])
    plank.sort(0)
    
    
    ############################ ストリンガー生成 ############################
    # 今回はデモなので、Y軸に並行にストリンガーを作成する
    y_st1_st = max(plank_l.getYfromX(x_st1))
    y_st2_st = min(plank_l.getYfromX(x_st2))
    y_st1_ed = max(plank_l.getYfromX(x_st1 + st_width))
    y_st2_ed = min(plank_l.getYfromX(x_st2 + st_width))
    st1 = clib.Polyline([x_st1, x_st1, x_st1 + st_width, x_st1 + st_width],\
                        [y_st1_st, y_st1_st-st_hight, y_st1_ed-st_hight, y_st1_ed])
    st2 = clib.Polyline([x_st2, x_st2, x_st2 + st_width, x_st2 + st_width],\
                        [y_st2_st, y_st2_st+st_hight, y_st2_ed+st_hight, y_st2_ed])
    
    ############################ リブ外形を整形 ############################
    rib_outer_u = clib.trim(outer, 0, u_outer_u)
    rib_outer_l = clib.trim(outer, u_outer_l, 1)
    if not outer.closed: # リブが閉じていない場合、後縁を整形
        line_edge = clib.SLine([outer.st[0], outer.ed[0]], [outer.st[1], outer.ed[1]])
        rib_outer = clib.LineGroup([rib_outer_u, plank_l, rib_outer_l, line_edge, line_plank_u, line_plank_l])
    else:
        rib_outer = clib.LineGroup([rib_outer_u, plank_l, rib_outer_l, line_plank_u, line_plank_l])
    rib_outer.sort(0)
        
    ############################ 桁穴生成 ################################
    cy = float(center.getYfromX(cx))
    circle = clib.Circle(cr, cx, cy)
    
    
    ############################ 迎角線生成 ##############################
    line_a0 = clib.SLine([-20, chord+20], [cy, cy])
    line_a1 = clib.rotate(line_a0, rot, cx, cy)
    
    
    ############################ トラス肉抜き　###############################
    # リブ外形からトラス幅だけオフセットしたスプラインを作成
    tr_u = clib.offset(upper, -tr_dist) # 上面
    tr_l = clib.offset(lower, tr_dist) # 下面
    # トラスのもととなる、桁中心を通るトラス角度の線l0を作成
    l0 = clib.SLine([cx - cr*np.cos(tr_angle), cx + cr*np.cos(tr_angle)],
                    [cy - cr*np.sin(tr_angle), cy + cr*np.sin(tr_angle)])
    # l0を桁径だけオフセット
    # 前方
    l0_f = clib.offset(l0, cr)
    l0_f = clib.invert(l0_f) # make_truss関数でのl2と向きを合わせる
    
    # 後方
    l0_r = clib.offset(l0, -cr)
    
    # トラス生成用関数
    def make_truss(l0, dist, angle, f0, f1):
        # l0を前方向にオフセットした線l1を作成
        l1 = clib.offset(l0, dist)
        # l1と上下面のスプラインの交点を算出
        u1, x1, y1 = clib.getCrossPointFromCurveLine(l1.f_line, f0.f_curve, 0.5) # 上面
        u2, x2, y2 = clib.getCrossPointFromCurveLine(l1.f_line, f1.f_curve, 0.5) # 下面
        # 算出した交点で、l1をトリム
        l1 = clib.trim(l1, x1, x2, lineAxis = "x") 
        
        # l1の端点（下面のスプラインとの交点）を通り、角度が-トラス角の線l2を作成
        l2 = clib.SLine([x2 - cr*np.cos(-angle), x2 + cr*np.cos(-angle)],
                        [y2 - cr*np.sin(-angle), y2 + cr*np.sin(-angle)])
        # l2と上面のスプラインの交点を算出
        u3, x3, y3 = clib.getCrossPointFromCurveLine(l2.f_line, f0.f_curve, 0.5) 
        # 算出した交点でl2をトリム
        l2 = clib.trim(l2, x2, x3, lineAxis = "x")
        
        # トラスの残りの１辺について、算出した交点をつなぐ線l3を算出
        l3 = clib.SLine([x3, x1], [y3, y1])
        
        # l1~l3をの間のフィレットを作成
        l1, l2, f_12 = clib.filetLines(l1, l2, fr)
        l3, l1, f_31 = clib.filetLines(l3, l1, fr)
        l2, l3, f_23 = clib.filetLines(l2, l3, fr)
        ls_tras = clib.LineGroup([l1, l2, l3, f_12, f_23, f_31])
        ls_tras.sort(0)
        
        # フィレットつきで面積を計算すると、フィレットできない面積となった場合に、肉抜き面積が破綻するため
        # フィレットなしの三角形で肉抜き停止を判定する
        area_triangle = clib.getAreaTriangle(x1,y1, x2,y2, x3,y3)
        
        return ls_tras, l2, area_triangle
    
    # リブ前方方向へ肉抜き作成
    ls_f_list = []
    i = 0
    area = area_min # 初期値なのでなんでもよい
    l2 = l0_f
    while area >= area_min:
        # 奇数のときと偶数のときでトラスの上下を入れ替える
        if i%2 == 0:
            ls_tras, l2, area = make_truss(l2, -tr_dist, tr_angle, tr_u, tr_l)
        else:
            ls_tras, l2, area = make_truss(l2, tr_dist, -tr_angle, tr_l, tr_u)
        ls_f_list.append(ls_tras)
        i += 1

    # リブ後方方向へ肉抜き作成
    ls_r_list = []
    i = 0
    area = area_min # 初期値なのでなんでもよい
    l2 = l0_r
    while area >= area_min:
        # 奇数のときと偶数のときでトラスの上下を入れ替える
        if i%2 == 0:
            ls_tras, l2, area = make_truss(l2, -tr_dist, tr_angle, tr_l, tr_u)
        else:
            ls_tras, l2, area = make_truss(l2, tr_dist, -tr_angle, tr_u, tr_l)
        ls_r_list.append(ls_tras)
        i += 1
    
    ############################ 面積を表示 ###############################
    ls_f_area_sum = 0
    for ls_f in ls_f_list:
        ls_f_area_sum += ls_f.area
    for ls_r in ls_r_list:
        ls_f_area_sum += ls_r.area
    
    # リブ面積算出
    # リブ外形 - 肉抜き面積 - 桁穴面積 - プランク面積 
    rib_area = rib_outer.area - ls_f_area_sum - circle.area - plank.area
    print("リブ面積: %s mm^2"%rib_area)
    print("プランク面積: %s mm^2"%plank.area)
    print("肉抜き面積: %s mm^2"%ls_f_area_sum)
    
    ############################ DXFへ出力 ################################
    # dxfファイルに保存
    # ezdxfのモデルワークスペースオブジェクトを生成
    doc = ez.new('R2010', setup=True)
    msp = doc.modelspace()
    
    # レイヤーを追加
    attr = {'color': 0, #色
            'lineweight':2, #線幅
            'linetype': 'Continuous' #線種
    }
    doc.layers.new(name="layer0",  dxfattribs = attr) # レイヤーを追加
    
    # モデルワークスペースに作成した直線とスプラインと楕円を追加
    clib.exportLine2ModeWorkSpace(msp, "layer0", rib_outer)
    clib.exportLine2ModeWorkSpace(msp, "layer0", center, color = 3, linetypes="CENTER")
    clib.exportLine2ModeWorkSpace(msp, "layer0", plank)
    clib.exportLine2ModeWorkSpace(msp, "layer0", line_plank_u)
    clib.exportLine2ModeWorkSpace(msp, "layer0", line_plank_l)
    clib.exportLine2ModeWorkSpace(msp, "layer0", st1)
    clib.exportLine2ModeWorkSpace(msp, "layer0", st2)
    clib.exportLine2ModeWorkSpace(msp, "layer0", circle)
    clib.exportLine2ModeWorkSpace(msp, "layer0", line_a0)
    clib.exportLine2ModeWorkSpace(msp, "layer0", line_a1, color = 1 , linetypes="CENTER")
    for ls_f in ls_f_list:
        clib.exportLine2ModeWorkSpace(msp, "layer0", ls_f)
    for ls_r in ls_r_list:
        clib.exportLine2ModeWorkSpace(msp, "layer0", ls_r)
    
    # dxfを出力
    doc.saveas('example_rib_generation.dxf')
    
    ############################ 終了 #################################
    
    
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        #plt.plot(outer.x, outer.y,"b")
        plt.plot(rib_outer.x, rib_outer.y ,"b", linewidth = 3)
        #plt.plot(upper.x, upper.y)
        #plt.plot(lower.x, lower.y)
        plt.plot(center.x, center.y, "g--")
        plt.plot(plank.x, plank.y, "r")
        plt.plot(st1.x, st1.y, "b")
        plt.plot(st2.x, st2.y, "b")
        plt.plot(circle.x, circle.y, "b")
        plt.plot(line_a0.x, line_a0.y, "b--")
        plt.plot(line_a1.x, line_a1.y, "r--")
        #plt.plot(tr_u.x, tr_u.y, "b--")
        #plt.plot(tr_l.x, tr_l.y, "b--")
        #plt.plot(l0_f.x, l0_f.y, "b--")
        
        for ls_f in ls_f_list:
            plt.plot(ls_f.x, ls_f.y, "b")

        for ls_r in ls_r_list:
            plt.plot(ls_r.x, ls_r.y, "b")
        
        plt.axis("equal")
        plt.title("Rib generation Example")
        plt.savefig("res/Example of Rib_generation.svg")


def example_3_2_1_1(plotGraph):
    #座標点変化量による接線の傾き算出
    m1 = clib.getM1(af.NACA2412_X, af.NACA2412_Y)
    
    #arctan2による接線の傾き算出
    sita1 = clib.getSita1(af.NACA2412_X, af.NACA2412_Y)
    m1_atan2 = np.tan(sita1)
    
    #グラフを描画
    if plotGraph == True:
        fig = plt.figure(figsize=(10.0, 8.0))
        ax1 = fig.add_subplot(111)
        ax1.plot(af.NACA2412_X, af.NACA2412_Y, "bo--")
        ax1.quiver(af.NACA2412_X, af.NACA2412_Y, np.cos(sita1), np.sin(sita1),\
                   angles='xy',scale_units='xy',scale=150, width=0.005, color = 'blue')
        ax2 = ax1.twinx()
        ax2.step(af.NACA2412_X, m1, "r")
        ax2.step(af.NACA2412_X, m1_atan2, "k--")
        ax2.set_ylim([-1,1])
        ax2.grid(True)
        ax1.legend(["RawData"],loc="lower right")
        ax2.legend(["Numerical Differentiation", "Arctan2"] ,loc="upper right")
        ax1.set_aspect("equal")
        plt.title("Example of 3.2.1.1")
        plt.show()
        plt.savefig("res/Example of 3.2.1.1.svg")
        
    return m1, sita1, m1_atan2 


def example_3_2_2_1(plotGraph):
    #座標点変化量による垂線の傾き算出
    m2 = clib.getM2(af.NACA2412_X, af.NACA2412_Y)
    
    #arctan2による垂線の傾き算出
    sita2 = clib.getSita2(af.NACA2412_X, af.NACA2412_Y)
    m2_atan2 = np.tan(sita2)
    
    #グラフを描画
    if plotGraph == True:
        fig = plt.figure(figsize=(10.0, 8.0))
        ax1 = fig.add_subplot(111)
        ax1.plot(af.NACA2412_X, af.NACA2412_Y, "bo--")
        ax1.quiver(af.NACA2412_X, af.NACA2412_Y, np.cos(sita2), np.sin(sita2), \
                   angles='xy',scale_units='xy',scale=50, width=0.005, color = 'blue')
        ax2 = ax1.twinx()
        ax2.step(af.NACA2412_X, m2, "r")
        ax2.step(af.NACA2412_X, m2_atan2, "k--")
        ax2.set_ylim([-500,500])
        ax2.grid(True)
        ax1.legend(["RawData"],loc="lower right")
        ax2.legend(["Numerical Differentiation", "Arctan2"] ,loc="upper right")      
        ax1.set_aspect("equal")
        plt.title("Example of 3.2.2.1")
        plt.show()
        plt.savefig("res/Example of 3.2.2.1.svg")
        
    return m2, sita2, m2_atan2     

def example_3_5_1(plotGraph):
    # 直線を作成
    l0 = clib.SLine([-3, 3], [1, 3])
    l1 = clib.SLine([-2, 4], [2, -3])
    
    #　交点を算出
    cx, cy = clib.getCrossPointFromLines(l0.a, l0.b, l1.a, l1.b)
    
    #　グラフを描画
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 3.5.1")
        plt.plot(l0.x, l0.y, "b-")
        plt.plot(l1.x, l1.y, "r-")
        plt.plot(cx, cy, "go")
        plt.axis("equal")
        plt.legend(["line0", "line1", "Cross Point"])
        plt.show()
        plt.savefig("res/Example of 3.5.1.svg")
    return l0, l1, cx, cy


def example_3_5_2(plotGraph):
    # 直線とスプラインを作成
    line = clib.SLine([0.1, 0.4], [-0.2, 0.3])
    spline = clib.Spline(af.NACA2412_X[0:52], af.NACA2412_Y[0:52])    
    
    #　交点を算出
    u_root, cx, cy = clib.getCrossPointFromCurveLine(line.f_line, spline.f_curve, 0.5)    

    #　グラフを描画
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 3.5.2")
        plt.plot(line.x, line.y, "b")
        plt.plot(spline.x, spline.y, "r")
        plt.plot(cx, cy, "go")
        plt.axis("equal")
        plt.legend(["line", "spline", "Cross Point"])
        plt.show()
        plt.savefig("res/Example of 3.5.2.svg")
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
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 3.5.3")
        plt.plot(spline1.x, spline1.y, "b")
        plt.plot(spline2.x, spline2.y, "r")
        plt.plot(cx1, cy1, "go")
        plt.plot(cx2, cy2, "ko")
        plt.axis("equal")  
        plt.legend(["spline1", "spline2", "Cross Point1", "Cross Point2"])
        plt.show()
        plt.savefig("res/Example of 3.5.3.svg")
    return  spline1, spline2, u_root1, s_root1


def example_3_5_4(plotGraph):
    # スプラインを作成
    spline = clib.Spline(af.NACA2412_X, af.NACA2412_Y)
    o_spline = clib.offset(spline, 0.03)

    #　交点を算出
    u_root0, s_root0, x_root0, y_root0 = clib.getCrossPointFromSelfCurve(o_spline.f_curve, 0, 1)
    u_root1, s_root1, x_root1, y_root1 = clib.getCrossPointFromSelfCurve(o_spline.f_curve, 0.45, 0.55)

    #　グラフを描画
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 3.5.3")
        plt.plot(spline.x, spline.y, "b")
        plt.plot(o_spline.x, o_spline.y, "r")
        plt.plot(x_root0, y_root0, "go")
        plt.plot(x_root1, y_root1, "ko")
        plt.axis("equal")  
        plt.legend(["spline", "offset spline", "Cross Point1", "Cross Point2"])
        plt.show()
        plt.savefig("res/Example of 3.5.4.svg")


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
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 3.6.1.1")
        plt.plot(x_s, y_s, "b")
        plt.plot(x_p, y_p, "k--")
        plt.plot(af.NACA2412_X, af.NACA2412_Y, "ro")
        plt.legend(["spline", "polyline", "RawData"])
        plt.axis("equal")
        plt.savefig("res/Example of 3.6.1.1_1.svg")
        
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 3.6.1.1")
        plt.plot(x_s, y_s, "bo-")
        plt.plot(x_p, y_p, "ko-")
        plt.plot(af.NACA2412_X, af.NACA2412_Y, "ro")
        plt.legend(["spline", "polyline", "RawData"])
        plt.axis("equal")
        plt.savefig("res/Example of 3.6.1.1_2.svg")
        plt.show()
    
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
        
        fig = plt.figure(figsize=(10.0, 8.0))
        ax1 = fig.add_subplot(111)
        ax1.plot(x_s, y_s, "b-")
        ax1.plot(x_p, y_p, "k--")

        ax2 = ax1.twinx()
        ax2.plot(x_s, m1_s, "b--")
        ax2.step(x_p, m1_p, "k--")
        ax2.set_ylim([-1,1])
        ax2.grid(True)
        ax1.plot(af.NACA2412_X, af.NACA2412_Y, "r")
        ax1.quiver(x_s, y_s, np.cos(sita_s), np.sin(sita_s), \
           angles='xy',scale_units='xy',scale=3000, width=0.003, color = 'blue')
        plt.legend(["spline", "polyline", "RawData"])
        ax1.set_aspect("equal")
        plt.title("Example of 3.6.1.2")
        plt.savefig("res/Example of 3.6.1.2.svg")
        plt.show()
    
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
        
        fig = plt.figure(figsize=(10.0, 8.0))
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
        plt.title("Example of 3.6.1.3")
        plt.savefig("res/Example of 3.6.1.3.svg")
        plt.show()
    
    return m2_s, m2_p


def example_3_6_2(plotGraph):
    airfoil = clib.Airfoil(af.NACA2412_X, af.NACA2412_Y)
    u1 = clib.getUCosine(500, airfoil.u_le)
    u2 = np.linspace(0,1,500)
    x1, y1 = airfoil.getPoint(u1)
    x2, y2 = airfoil.getPoint(u2)

    #グラフを描画
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0)) 
        plt.plot(u2, u2, "b")
        plt.plot(u2, u1, "r")
        plt.legend(["Linear", "Cosine"])
        plt.axis("equal") 
        plt.title("Example of 3.6.2")
        plt.savefig("res/Example of 3.6.2_1.svg")

        plt.figure(figsize=(10.0, 8.0)) 
        plt.plot(x1, y1, "bo--")
        plt.plot(x2, y2, "ro--")
        plt.legend(["Linear", "Cosine"])
        plt.axis("equal") 
        plt.title("Example of 3.6.2")
        plt.savefig("res/Example of 3.6.2_2.svg")

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
        plt.figure(figsize=(10.0, 8.0))
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
        plt.savefig("res/Example of 3.7.svg")
        plt.show()        


def example_3_9_1(plotGraph):
    # 翼形座標データを読み込み
    #x, y = clib.importFromText("foils/naca2412.dat", ' ', 1)
    x, y = clib.importFromText("foils/dae51.dat", '\t', 1)    
    
    # 翼形クラスのインスタンスを作成
    airfoil = clib.Airfoil(x, y)
    
    # x軸方向のコサイン補完点列を作成
    x_func = clib.getXCosine(airfoil.xmin, airfoil.xmax, 300)
    uy = airfoil.f_upper(x_func)
    ly = airfoil.f_lower(x_func)
    
    # グラフを描画
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 3.9.1")
        
        #plt.plot(airfoil.x_intp, airfoil.y_intp, "b-")
        plt.plot(x_func, uy, "r")
        plt.plot(x_func, ly, "b")
        #plt.plot(x, y, "ro")
        
        plt.axis("equal")
        plt.legend(["Upper func", "lower func"])
        plt.savefig("res/Example of 3.9.1.svg")
        plt.show()    


def example_3_9_2(plotGraph):
    # 翼形座標データを読み込み
    x, y = clib.importFromText("foils/naca2412.dat", ' ', 1)
    #x, y = clib.importFromText("foils/dae51.dat", '\t', 1)    
    
    # 翼形クラスのインスタンスを作成
    airfoil = clib.Airfoil(x, y)
    
    # x軸方向のコサイン補完点列を作成
    x_func = clib.getXCosine(airfoil.xmin, airfoil.xmax, 300)
    
    # 中心線のy座標を取得
    cy = airfoil.f_center(x_func)
    
    # グラフを描画
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 3.9.2")
        plt.plot(airfoil.x_intp, airfoil.y_intp, "b")
        plt.plot(x_func, cy, "g")
        plt.axis("equal")
        plt.legend(["Spline", "Camber line"])
        plt.savefig("res/Example of 3.9.2.svg")
        plt.show()    


def example_3_9_3(plotGraph):
    # DAE51, NACA2412、およびそれらをXFLRで20%混合した座標点を読み込み
    x1, y1 = clib.importFromText("foils/dae51.dat", '\t', 1)
    x2, y2 = clib.importFromText("foils/naca2412.dat", ' ', 1)
    xm, ym = clib.importFromText("foils/dae_naca_poor.dat", '    ', 1)
    
    # DAE51とNACA2412の翼形クラスのインスタンスを作成
    dae51 = clib.Airfoil(x1, y1)
    naca2412 = clib.Airfoil(x2, y2)
    
    # DAE51とNACA2412を20%混合した座標点を作成
    mix_airfoil, x = clib.mixAirfoil(dae51, naca2412, 0.2, True) 
    mix_airfoil_xflr, naca2412_p = clib.mixAirfoilXFLR(dae51, naca2412, 0.2, True) 
    
    dae_uy = dae51.f_upper(x)
    dae_ly = dae51.f_lower(x)
    dae_y = np.concatenate([dae_uy[-1::-1], dae_ly], 0)
    
    naca_uy = naca2412.f_upper(x)
    naca_ly = naca2412.f_lower(x)
    naca_y = np.concatenate([naca_uy[-1::-1], naca_ly], 0)
    x_foil = np.concatenate([x[-1::-1], x], 0)

    # グラフを描画
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 3.9.3 (X Method)")
        plt.plot(dae51.x_intp, dae51.y_intp, "g--")
        plt.plot(naca2412.x_intp, naca2412.y_intp, "b--")
        plt.plot(mix_airfoil.x_intp, mix_airfoil.y_intp, "c--")
        plt.plot(xm, ym, "ro")
        plt.plot(x_foil, dae_y, "go")
        plt.plot(x_foil, naca_y, "bo")
        plt.plot(mix_airfoil.x, mix_airfoil.y, "co")
        plt.axis("equal")
        plt.legend(["DAE51", "NACA2412", "Mix(by X Method)","XFLR Mix"])
        plt.savefig("res/Example of 3.9.3_1.svg")
        plt.show()    
        
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 3.9.3 (XFLR Method)")
        plt.plot(dae51.x_intp, dae51.y_intp, "g--")
        plt.plot(naca2412.x_intp, naca2412.y_intp, "b--")
        plt.plot(mix_airfoil_xflr.x_intp, mix_airfoil_xflr.y_intp, "c--")
        plt.plot(xm, ym, "ro")
        plt.plot(x1, y1, "go")
        plt.plot(naca2412_p.x, naca2412_p.y, "bo")
        plt.plot(mix_airfoil_xflr.x, mix_airfoil_xflr.y, "co")
        plt.axis("equal")
        plt.legend(["DAE51", "NACA2412", "Mix(by XFLR Method)","XFLR Mix"])
        plt.savefig("res/Example of 3.9.3_2.svg")
        plt.show()    
    

def example_4_1(plotGraph):
    # ファイルから座標点列を読み込み
    x, y  = clib.importFromText("NACA2412.csv", ",", 1)
    line = clib.Spline(x,y)
    
    # グラフを描画
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 4.1")
        plt.plot(line.x, line.y, "bo-")
        plt.axis("equal")
        plt.legend(["Raw data(from csv file)"])
        plt.savefig("res/Example of 4.1.svg")
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
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 4.2")
        plt.plot(line.x, line.y, "b--")
        plt.plot(spline.x, spline.y, "b--")
        plt.plot(ellipse.x, ellipse.y, "b--")
        plt.plot(m_line.x, m_line.y, "r")
        plt.plot(m_spline.x, m_spline.y, "r")
        plt.plot(m_ellipse.x, m_ellipse.y, "r")
        plt.axis("equal")
        plt.legend(["Line", "Spline", "Ellipese", \
                    "Moved line", "Moved spline", "Moved ellipse"])
        plt.savefig("res/Example of 4.2.svg")
        plt.show()


def example_4_3(plotGraph):
    # 直線とスプラインと楕円を作成
    line = clib.SLine([0.1, 0.4], [-0.2, 0.3])
    spline = clib.Spline(af.NACA2412_X, af.NACA2412_Y)
    ellipse = clib.Ellipse(0.5, 0.25, 0, 1, 1)
    
    # (x,y) = (0.5, 0.5)を中心に時計回りに45度回転する
    rx = 0.5
    ry = 0.5
    sita = np.radians(30)
    
    # 移動後の直線とスプラインを作成
    m_line = clib.rotate(line, sita, rx, ry)
    m_spline = clib.rotate(spline, sita, rx, ry)
    m_ellipse = clib.rotate(ellipse, sita, rx, ry)
    
    # グラフを描画
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 4.3")
        plt.plot(line.x, line.y, "b--")
        plt.plot(spline.x, spline.y, "b--")
        plt.plot(ellipse.x, ellipse.y, "b--")
        plt.plot(m_line.x, m_line.y, "r")
        plt.plot(m_spline.x, m_spline.y, "r")
        plt.plot(m_ellipse.x, m_ellipse.y, "r")
        plt.plot(rx, ry, "ro")
        plt.axis("equal")
        plt.legend(["Line", "Spline", "Ellipese", \
                    "Rotated line", "Rotated spline", "Rotated ellipse", "Rotation Center"])
        plt.savefig("res/Example of 4.3.svg")
        plt.show()    


def example_4_4(plotGraph):
    # 直線を作成
    l0 = clib.SLine([-3, 3], [-3, 3])
    l1 = clib.SLine([-3, 3], [3, -3])
    
    print("line0: a=%s, b=%s, sita=%s deg"%(l0.a, l0.b, np.degrees(l0.sita)))
    print("line1: a=%s, b=%s, sita=%s deg"%(l1.a, l1.b, np.degrees(l1.sita)))
    
    #　グラフを描画
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 4.4")
        plt.plot(l0.x, l0.y, "b")
        plt.plot(l1.x, l1.y, "r")
        plt.axis("equal")  
        plt.grid(True)
        plt.legend(["Line0", "Line1"])
        plt.savefig("res/Example of 4.4.svg")
        plt.show()
    return l0, l1


def example_4_6(plotGraph):
    #半径2, 中心(1,1)とした円を作成
    r = 2
    rx = 1
    ry = 1
    circle = clib.Circle(r, rx, ry)
    

    sita_st = 0 
    sita_ed = 90
    arc = clib.Arc(r, rx, ry, np.radians(sita_st), np.radians(sita_ed))
    
    #補完点を200点生成。
    interp_point = np.linspace(0,1,200)  
    x_i, y_i = circle.getPoint(interp_point)

    #　グラフを描画
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 4.6")
        plt.plot(circle.x, circle.y, "b")
        plt.plot(arc.x, arc.y, "r--")
        #plt.plot(x_i, y_i, "r--")
        plt.axis("equal")  
        plt.grid(True)
        plt.legend(["Circle", "Arc"])
        plt.savefig("res/Example of 4.6.svg")
        plt.show()


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
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 4.7")
        #plt.plot(ellipse.x, ellipse.y, "bo")
        plt.plot(x_i, y_i, "r")
        plt.axis("equal")  
        plt.grid(True)
        plt.legend(["Ellipse", "Interpolated Ellipse"])
        plt.savefig("res/Example of 4.7.svg")
        plt.show()


def example_4_8_1(plotGraph):
    # 直線とスプラインと円を作成
    line = clib.SLine([0.1, 0.4], [-0.2, 0.3])
    spline = clib.Spline(af.NACA2412_X, af.NACA2412_Y)
    circle = clib.Circle(0.1, 0.5, 0)
    
    # 0.01だけオフセットする
    d = 0.01
    
    # 外側にオフセット後の直線とスプラインを作成
    o_line = clib.offset(line, d)
    o_spline = clib.offset(spline, -d)
    o_circle = clib.offset(circle, -d)
    
    # 内側にオフセットしたスプラインを作成
    d1 = 0.03
    o_spline_self_col = clib.offset(spline, d1)
    u0_root, u1_root, x_root, y_root = clib.getCrossPointFromSelfCurve(o_spline_self_col.f_curve, 0, 1)
    t_spline = clib.trim(o_spline_self_col, u0_root, u1_root)
    tr_spline = clib.removeSelfCollision(t_spline)
    
    # グラフを描画
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 4.8.1")
        plt.plot(line.x, line.y, "b--")
        plt.plot(spline.x, spline.y, "b--")
        plt.plot(circle.x, circle.y, "b--")
        plt.plot(o_line.x, o_line.y, "r")
        plt.plot(o_spline.x, o_spline.y, "r")
        plt.plot(o_circle.x, o_circle.y, "r")
        plt.axis("equal")
        plt.legend(["Line", "Spline", "Circle", \
                    "Offseted line", "Offseted spline", "Offseted circle"])
        plt.savefig("res/Example of 4.8.1_1.svg")  

        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 4.8.1")
        plt.plot(spline.x, spline.y, "b--")
        plt.plot(o_spline_self_col.x, o_spline_self_col.y, "r")
        plt.axis("equal")
        plt.legend(["Spline", "Offseted spline"])
        plt.savefig("res/Example of 4.8.1_2.svg")  

        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 4.8.1")
        plt.plot(spline.x, spline.y, "b--")
        plt.plot(t_spline.x, t_spline.y, "r")
        plt.axis("equal")
        plt.legend(["Spline", "Offseted spline"])
        plt.savefig("res/Example of 4.8.1_3.svg")  

        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 4.8.1")
        plt.plot(spline.x, spline.y, "b--")
        plt.plot(tr_spline.x, tr_spline.y, "r")
        plt.axis("equal")
        plt.legend(["Spline", "Offseted spline"])
        plt.savefig("res/Example of 4.8.1_4.svg")  



def example_4_8_2(plotGraph):
    airfoil = clib.Airfoil(af.NACA2412_X, af.NACA2412_Y)
    airfoil = clib.trim(airfoil, 0.25, 0.75)
    airfoil = clib.scale(airfoil, 300, 0, 0)
    poly_airfoil = clib.convert2Polyline(airfoil, 500)
    collision_airfoil = clib.offset(airfoil, 10)
    fixed_airfoil = clib.offset(airfoil, 10)
    fixed_airfoil = clib.removeSelfCollision(fixed_airfoil)
    fixed_poly_airfoil = clib.offset(poly_airfoil, 10)
    fixed_poly_airfoil = clib.removeSelfCollision(fixed_poly_airfoil)
    
    fixed_airfoil.setIntporatePoints(np.linspace(0,1, 1000))
    fixed_poly_airfoil.setIntporatePoints(np.linspace(0,1, 1000))
    
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 4.8.2")
        plt.plot(airfoil.x, airfoil.y, "b--")
        plt.plot(collision_airfoil.x, collision_airfoil.y, "b")
        plt.plot(fixed_airfoil.x_intp, fixed_airfoil.y_intp, "r--", linewidth = 2)
        plt.plot(fixed_poly_airfoil.x_intp, fixed_poly_airfoil.y_intp, "g--")
        plt.legend(["Before Offset","Collision caused by Offset", "Collision fixed (Spline)", "Collision fixed (Polyline)"])
        plt.axis("equal")  
        plt.savefig("res/Example of 4.8.2.svg")  
        plt.show()


def example_4_9_2(plotGraph):
    l0 = clib.SLine([-3,-1], [5,5])
    l1 = clib.SLine([-1,-2], [5.5,6])
    
    new_l0, new_l1, filet = clib.filetLines(l0, l1, 0.2)
    new_l0_j, new_l1_j, filet_j = clib.filetLines(l0, l1, 0.2, True)
    
    #グラフを描画
    if plotGraph == True:  
        plt.figure(figsize=(10.0, 8.0))
        plt.plot(l0.x, l0.y, "r")
        plt.plot(l1.x, l1.y, "b")
        plt.plot(filet.x, filet.y, "g")
        plt.plot(filet.cx, filet.cy, "go")
        plt.plot(new_l0.x, new_l0.y, "ro--")
        plt.plot(new_l1.x, new_l1.y, "bo--")
        #plt.plot(filet_j.x, filet_j.y, "k")
        plt.axis("equal")    
        plt.title("Example of 4.9.2")
        plt.legend(["line0","line1", "filet", "filet centor"])
        plt.savefig("res/Example of 4.9.2.svg")  
        plt.show()


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
        plt.figure(figsize=(10.0, 8.0))
        plt.plot(line.x, line.y, "b")
        plt.plot(spline.x, spline.y, "b")
        plt.plot(filet.x, filet.y, "r--")
        plt.plot(t_spline.x, t_spline.y, "r--")
        plt.plot(t_line.x, t_line.y, "r--")
        plt.axis("equal")
        plt.title("Example of 4.9.3")
        plt.savefig("res/Example of 4.9.3_1.svg") 
        plt.show()
    
    
def example_4_9_4(plotGraph):
    # 交差するスプラインを2つ作成
    spline1 = clib.Spline(af.NACA2412_X[20:52], af.NACA2412_Y[20:52] - 0.03)
    spline2 = clib.Spline(af.NACA2412_X[51:-20], af.NACA2412_Y[51:-20] + 0.03)

    # スプラインどうしの交点を、r=0.005でフィレットする。
    # フィレット箇所は、スプライン1に対して第一象限とする
    # 複数交点を持つ場合に備え、探索始点はスプライン1、スプライン2の10%位置とする    
    t_spline1, t_spline2, filet = clib.filetCurves(spline1, spline2, 0.005, 1, 0.4, 0.4)
    
    #グラフを描画
    if plotGraph == True:    
        plt.figure(figsize=(10.0, 8.0))
        plt.plot(spline1.x, spline1.y, "b")
        plt.plot(spline2.x, spline2.y, "b")
        plt.plot(filet.x, filet.y, "r--")
        plt.plot(t_spline1.x, t_spline1.y, "r--")
        plt.plot(t_spline2.x, t_spline2.y, "r--")        
        plt.axis("equal")
        plt.title("Example of 4.9.4")
        plt.savefig("res/Example of 4.9.4_1.svg") 
        plt.show()
    
  
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
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 4.10")
        plt.plot(x0, y0, "go")
        plt.plot(line.x, line.y, "b--")
        plt.plot(spline.x, spline.y, "b--")
        plt.plot(circle.x, circle.y, "b--")
        plt.plot(s_line.x, s_line.y, "r")
        plt.plot(s_spline.x, s_spline.y, "r")
        plt.plot(s_circle.x, s_circle.y, "r")
        plt.axis("equal")
        plt.legend(["Scaled Center","Line", "Spline", "Circle", \
                    "Scaled line", "Scaled spline", "Scaled circle"])
        plt.savefig("res/Example of 4.10.svg") 
        plt.show()        


def example_4_11_1(plotGraph):
    # 線分を作成
    line = clib.SLine([-3, 2], [1, 5])

    # x:-1~0の範囲でトリム
    x_st = -1
    x_ed = 0
    xt_line = clib.trim(line, x_st, x_ed, "x")
    
    # y:4~4.5の範囲でトリム
    y_st = 4
    y_ed = 4.5
    yt_line = clib.trim(line, y_st, y_ed, "y")
    
    # グラフを描画
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 4.11.1")
        plt.plot(line.x, line.y, "b")
        plt.plot(xt_line.x, xt_line.y, "r--", linewidth = 3)
        plt.plot(yt_line.x, yt_line.y, "g--", linewidth = 3)
        plt.axis("equal")
        plt.legend(["Original Line", "Trimed Line by X", "Trimed Line by Y"])
        
        plt.plot([x_st, x_st], line.y, "k--")
        plt.plot([x_ed, x_ed], line.y, "k--")
        plt.plot(line.x, [y_st, y_st],  "k--")
        plt.plot(line.x, [y_ed, y_ed],  "k--")
        plt.savefig("res/Example of 4.11.1.svg") 
        plt.show()         


def example_4_11_2(plotGraph):
    # 交差するスプラインを2つ作成
    spline1 = clib.Spline(af.NACA2412_X[0:52], af.NACA2412_Y[0:52] - 0.03)
    spline2 = clib.Spline(af.NACA2412_X[53:-1], af.NACA2412_Y[53:-1] + 0.03)
    airfoil = clib.Airfoil(af.NACA2412_X, af.NACA2412_Y)
    
    #　交点を算出
    u_root1, s_root1, cx1, cy1 = clib.getCrossPointFromCurves(spline1.f_curve, spline2.f_curve, 0.7, 0.1)
    u_root2, s_root2, cx2, cy2 = clib.getCrossPointFromCurves(spline1.f_curve, spline2.f_curve, 0.1, 0.7)

    # x = 0.5でトリムしたスプラインを作成
    u_trim = airfoil.getUfromX(0.5)

    # 交点でトリムしたスプラインを作成
    u_st = 0
    u_ed = u_root1
    s_st = s_root1
    s_ed = 1
    t_spline1 = clib.trim(spline1, u_st, u_ed)
    t_spline2 = clib.trim(spline2, s_st, s_ed)
    t_airfoil = clib.trim(airfoil, u_trim[0], u_trim[1])
    
    # グラフを描画
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 4.11.2")
        plt.plot(spline1.x, spline1.y, "b--")
        plt.plot(spline2.x, spline2.y, "b--")
        plt.plot(airfoil.x, airfoil.y, "b--")
        plt.plot(t_spline1.x, t_spline1.y, "r")
        plt.plot(t_spline2.x, t_spline2.y, "r")
        plt.plot(t_airfoil.x, t_airfoil.y, "g")
        plt.plot([0.5, 0.5], [-0.1, 0.1], "g--")
        plt.axis("equal")
        plt.savefig("res/Example of 4.11.2.svg") 
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
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 4.11.4")
        plt.plot(circle.x, circle.y, "b--")
        plt.plot(t_arc.x, t_arc.y, "r")
        plt.axis("equal")
        plt.legend(["Circle", "Trimed Circle"])
        plt.savefig("res/Example of 4.11.3.svg") 
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
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 4.11.4")
        plt.plot(ellipse.x, ellipse.y, "b--")
        plt.plot(t_spline.x, t_spline.y, "r")
        plt.axis("equal")
        plt.legend(["Ellipse", "Trimed Ellipse"])
        plt.savefig("res/Example of 4.11.4.svg") 
        plt.show()       
        
        
def example_4_12_1(plotGraph):
    # 直線とスプラインと円弧と楕円を作成
    line = clib.SLine([0.1, 0.4], [-0.2, 0.3])
    spline = clib.Spline(af.NACA2412_X, af.NACA2412_Y)
    arc = clib.Arc(1, -0.5, -0.5, np.radians(20), np.radians(60))
    ellipse = clib.Ellipse(0.5, 0.25, 0, 1, 1)
    circle = clib.Circle(0.5, -1, -1)
    polyline = clib.Polyline(af.NACA2412_X, af.NACA2412_Y)
    polyline = clib.move(polyline, 0, -1)
    
    # スクリプト出力用の文字列を作成
    scr = ""
    scr += clib.exportLine2CommandScript(line)
    scr += clib.exportLine2CommandScript(spline)
    scr += clib.exportLine2CommandScript(circle)
    scr += clib.exportLine2CommandScript(arc)
    scr += clib.exportLine2CommandScript(polyline)
    scr += clib.exportLine2CommandScript(ellipse)
    
    
    
    # スクリプト出力用の文字列を確認
    print("AutoCAD Script file is bellow")
    print(scr)
    
    # スクリプトファイル（.scr）として保存
    f = open("example_4_12_1.scr", "w")
    f.writelines(scr)
    f.close()
    
    # グラフを描画
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        plt.title("Example of 4.12.1")
        plt.plot(line.x, line.y, "b")
        plt.plot(spline.x, spline.y, "b")
        plt.plot(arc.x, arc.y, "b")
        plt.plot(ellipse.x, ellipse.y, "b")
        plt.plot(polyline.x, polyline.y, "b")
        plt.plot(circle.x, circle.y, "b")
        plt.axis("equal")
        plt.savefig("res/Example of 4.12.1.svg") 
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
        plt.title("Example of 4.12.2")
        plt.savefig("res/Example of 4.12.2.svg") 
        fig.show()        


def example_5_1(plotGraph):
    #doc = ez.readfile("test.dxf")
    doc = ez.readfile("example_4_12_2.dxf")
    msp = doc.modelspace()
    lines = clib.importLinesFromDxf(msp, "LINE")
    splines = clib.importLinesFromDxf(msp, "SPLINE")
    
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        for line in lines:
            plt.plot(line.x, line.y, "b")
        for spline in splines:
            plt.plot(spline.x, spline.y, "r")
        plt.axis("equal")
        plt.title("Example of 5.1")
        plt.savefig("res/Example of 5.1.svg") 
        plt.show()
    

def example_5_2_1(plotGraph):
    doc = ez.readfile("test.dxf")
    #doc = ez.readfile("example_4_12_2.dxf")
    msp = doc.modelspace()
    lines = clib.importLinesFromDxf(msp, "LINE")
    splines = clib.importLinesFromDxf(msp, "SPLINE")
    
    all_lines = lines + splines
    line_group_list = clib.detectLineGroups(all_lines, 2)
    
    i = 0
    while i < len(line_group_list):
        line_group = line_group_list[i]
        if line_group.closed == True:
            print("line group No.%s is closed"%i)
        else:
            print("line group No.%s is opened"%i)
        i += 1
    
    colors = ["b", "g", "r", "c", "m", "y", "k", "b--", "g--"]
    colors_st = ["bo", "go", "ro", "co", "mo", "yo", "ko", "bo", "go"]

    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        i = 0
        for line in all_lines:
            plt.plot(line.x, line.y, colors[i])
            plt.plot(line.st[0], line.st[1], colors_st[i])
            x0,y0,x1,y1 = getQuiver(line)
            plt.quiver(x0,y0,x1,y1, \
               angles='xy',scale_units='xy',scale=0.1, width=0.003, color = 'black')
            i += 1
            
        plt.axis("equal")
        plt.title("Example of 5.2.1 (Before sort)")
        plt.savefig("res/Example of 5.2.1_1.svg")
        plt.show()
        
        
        plt.figure(figsize=(10.0, 8.0))
        i = 0
        while i < len(line_group_list):
            line_group = line_group_list[i]
            init_line = line_group_list[i].lines[0]
            plt.plot(init_line.st[0], init_line.st[1], colors_st[i])                
            for line in line_group.lines:
                plt.plot(line.x, line.y, colors[i])
                x0,y0,x1,y1 = getQuiver(line)
                plt.quiver(x0,y0,x1,y1, \
                   angles='xy',scale_units='xy',scale=0.1, width=0.003, color = 'black')
            i += 1
        plt.axis("equal")
        plt.title("Example of 5.2.1 (After sort)")
        plt.savefig("res/Example of 5.2.1_2.svg")
        plt.show()
    return line_group_list


def example_5_2_2(plotGraph):
    line_group_list = example_5_2_1(False)
    
    i = 0
    while i < len(line_group_list):
        line_group = line_group_list[i]
        if line_group.ccw == True:
            print("line group No.%s direction is CCW"%i)
        else:
            print("line group No.%s direction is CW"%i)
        i += 1
    
    i = 0
    while i < len(line_group_list):
        line_group = line_group_list[i]
        if line_group.ccw == False:
            i_line_group = clib.invert(line_group)
            line_group_list[i] = i_line_group
            print("line group No.%s chenge direction to CCW"%i)
        if line_group.ccw == True:
            print("line group No.%s direction is CCW"%i)
        else:
            print("line group No.%s direction is CW"%i)
        i += 1
        
    colors = ["b", "g", "r", "c", "m", "y", "k", "b--", "g--"]
    colors_st = ["bo", "go", "ro", "co", "mo", "yo", "ko", "bo", "go"]
    
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        i = 0
        while i < len(line_group_list):
            line_group = line_group_list[i]
            init_line = line_group_list[i].lines[0]
            plt.plot(init_line.st[0], init_line.st[1], colors_st[i])                
            for line in line_group.lines:
                plt.plot(line.x, line.y, colors[i])
                x0,y0,x1,y1 = getQuiver(line)
                plt.quiver(x0,y0,x1,y1, \
                   angles='xy',scale_units='xy',scale=0.1, width=0.003, color = 'black')
            i += 1
        plt.title("Example of 5.2.2")
        plt.savefig("res/Example of 5.2_2.svg")
        plt.axis("equal")
        plt.show()


def example_5_3_1_1(plotGraph):
    airfoil = clib.Airfoil(af.NACA2412_X, af.NACA2412_Y)
    camber = clib.Spline(airfoil.cx, airfoil.cy)
    
    airfoil = clib.scale(airfoil, 300, 0, 0)
    camber = clib.scale(camber, 300, 0, 0)

    circle1 = clib.Circle(10, 100, camber.getYfromX(100))
    circle2 = clib.Circle(210, 100, camber.getYfromX(100))
    circle3 = clib.Circle(30, 100, camber.getYfromX(100))

    is_inside1 = clib.checkInclusion(airfoil, circle1)
    is_inside2 = clib.checkInclusion(airfoil, circle2)
    is_inside3 = clib.checkInclusion(airfoil, circle3)
    
    if is_inside1 == 0:
        print("Airfoil in Circle1")
    elif is_inside1 == 1:
        print("Circle1 in Airfoil")
    else:
        print("Circle1 is colliding with Airfoil")
        
    if is_inside2 == 0:
        print("Airfoil in Circle2")
    elif is_inside2 == 1:
        print("Circle2 in Airfoil")
    else:
        print("Circle2 is colliding with Airfoil")   
    
    if is_inside3 == 0:
        print("Airfoil in Circle3")
    elif is_inside3 == 1:
        print("Circle3 in Airfoil")
    else:
        print("Circle3 is colliding with Airfoil")   
    
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        x1_i, y1_i, x1_o, y1_o = clib.getInclusionList(airfoil, circle1)
        x2_i, y2_i, x2_o, y2_o = clib.getInclusionList(airfoil, circle2)
        x3_i, y3_i, x3_o, y3_o = clib.getInclusionList(airfoil, circle3)
        plt.plot(airfoil.x, airfoil.y, "b")
        plt.plot(circle1.x, circle1.y, "b")
        plt.plot(circle2.x, circle2.y, "b")
        plt.plot(circle3.x, circle3.y, "b")
        
        i = 0
        while i < len(x1_i):
            plt.plot(x1_i[i], y1_i[i], "c--", linewidth = 3)
            i += 1
        i = 0
        while i < len(x1_o):
            plt.plot(x1_o[i], y1_o[i], "r--", linewidth = 3)
            i += 1    
        i = 0
        while i < len(x2_i):
            plt.plot(x2_i[i], y2_i[i], "c--", linewidth = 3)
            i += 1
        i = 0
        while i < len(x2_o):
            plt.plot(x2_o[i], y2_o[i], "r--", linewidth = 3)
            i += 1
        i = 0
        while i < len(x3_i):
            plt.plot(x3_i[i], y3_i[i], "c--", linewidth = 3)
            i += 1
        i = 0
        while i < len(x3_o):
            plt.plot(x3_o[i], y3_o[i], "r--", linewidth = 3)
            i += 1                
        plt.title("Example of 5.3.1.1")
        plt.axis("equal")    
        plt.savefig("res/Example of 5.3.1_1.svg")
        plt.show()


def example_5_3_1_2(plotGraph):
    airfoil = clib.Airfoil(af.NACA2412_X, af.NACA2412_Y)
    airfoil = clib.scale(airfoil, 300, 0, 0)
    r = 10;
    cx = 100
    circle = clib.Circle(r, cx, 5)
    
    circle_is_inside = clib.checkInclusion(airfoil, circle)
    airfoil_is_inside = clib.checkInclusion(circle, airfoil)
    # 外側にあるオブジェクトは反時計回りに、内側にあるオブジェクトは時計回りにカットする
    if circle_is_inside == 0:
        if circle.ccw == False:
            c_circle = clib.invert(circle)
        else:
            c_circle = circle
    elif circle_is_inside == 1:
        if circle.ccw == True:
            c_circle = clib.invert(circle)
        else:
            c_circle = circle
    else:
        print("Collision detect!")

    if airfoil_is_inside == 0:
        if airfoil.ccw == False:
            c_airfoil = clib.invert(airfoil)
        else:
            c_airfoil = airfoil
    elif airfoil_is_inside == 1:
        if airfoil.ccw == True:
            c_airfoil = clib.invert(airfoil)
        else:
            c_airfoil = airfoil            
    else:
        print("Collision detect!")
        
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        plt.plot(airfoil.x, airfoil.y, "b")
        plt.plot(circle.x, circle.y, "b")
        x0,y0,x1,y1 = getQuiver(airfoil)
        plt.quiver(x0,y0,x1,y1, \
                   angles='xy',scale_units='xy',scale=0.04, width=0.006, color = 'red')
        
        x0,y0,x1,y1 = getQuiver(circle)
        plt.quiver(x0,y0,x1,y1, \
                   angles='xy',scale_units='xy',scale=0.04, width=0.006, color = 'red')
        plt.axis("equal")
        plt.title("Example of 5.3.1.2 (Before invert)")
        plt.savefig("res/Example of 5.3.1_2_1.svg")
        plt.show()

        plt.figure(figsize=(10.0, 8.0))
        plt.plot(c_airfoil.x, c_airfoil.y, "b")
        plt.plot(c_circle.x, c_circle.y, "b")
        x0,y0,x1,y1 = getQuiver(c_airfoil)
        plt.quiver(x0,y0,x1,y1, \
                   angles='xy',scale_units='xy',scale=0.04, width=0.006, color = 'red')
        
        x0,y0,x1,y1 = getQuiver(c_circle)
        plt.quiver(x0,y0,x1,y1, \
                   angles='xy',scale_units='xy',scale=0.04, width=0.006, color = 'red')
        plt.axis("equal")
        plt.title("Example of 5.3.1.2 (After invert)")
        plt.savefig("res/Example of 5.3.1_2_2.svg")
        plt.show()



def example_5_3_2(plotGraph):
    airfoil = clib.Airfoil(af.NACA2412_X, af.NACA2412_Y)
    airfoil = clib.scale(airfoil, 300, 0, 0)
    r = 10;
    cx = 100
    circle = clib.Circle(r, cx, 5)
    
    circle_is_inside = clib.checkInclusion(airfoil, circle)
    airfoil_is_inside = clib.checkInclusion(circle, airfoil)
    # 外側にあるオブジェクトは反時計回りに、内側にあるオブジェクトは時計回りにカットする
    # 外側にあるオブジェクトは外側にオフセット、内側にあるオブジェクトは内側にオフセットする
    # オフセット方向は、CCWの場合は逆向きにオフセットされるので、補正する
    d = 3
    if circle_is_inside == 0:
        if circle.ccw == False:
            circle = clib.invert(circle)
            o_circle = clib.offset(circle,d)
        else:
            o_circle = clib.offset(circle,-d)
    elif circle_is_inside == 1:
        if circle.ccw == True:
            circle = clib.invert(circle)
            o_circle = clib.offset(circle,-d)
        else:
            o_circle = clib.offset(circle,d)
    else:
        print("Collision detect!")

    if airfoil_is_inside == 0:
        if airfoil.ccw == False:
            airfoil = clib.invert(airfoil)
            o_airfoil = clib.offset(airfoil,d)
        else:
            o_airfoil = clib.offset(airfoil,-d)
    elif airfoil_is_inside == 1:
        if airfoil.ccw == True:
            airfoil = clib.invert(airfoil)
            o_airfoil = clib.offset(airfoil,-d)
        else:
            o_airfoil = clib.offset(airfoil,d)
    else:
        print("Collision detect!")
    
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        plt.plot(airfoil.x, airfoil.y, "b")
        plt.plot(o_airfoil.x, o_airfoil.y, "b--")
        plt.plot(circle.x, circle.y, "b")
        plt.plot(o_circle.x, o_circle.y, "b--")
        x0,y0,x1,y1 = getQuiver(airfoil)
        plt.quiver(x0,y0,x1,y1, \
                   angles='xy',scale_units='xy',scale=0.04, width=0.006, color = 'red')
        
        x0,y0,x1,y1 = getQuiver(circle)
        plt.quiver(x0,y0,x1,y1, \
                   angles='xy',scale_units='xy',scale=0.04, width=0.006, color = 'red')
        plt.axis("equal")  
        plt.title("Example of 5.3.2")
        plt.legend(["Original Line", "Offset Line"])
        plt.savefig("res/Example of 5.3.2_2.svg")
        plt.show()

    
def example_5_3_3(plotGraph):
    doc = ez.readfile("test1.dxf")
    msp = doc.modelspace()
    lines = clib.importLinesFromDxf(msp, "LINE")
    splines = clib.importLinesFromDxf(msp, "SPLINE")

    all_lines = lines + splines
    line_group_list = clib.detectLineGroups(all_lines, 2)
    d = 2
    
    i = 0
    o_line_group_list = []
    for line_group in line_group_list:
        if line_group.ccw == False:
            line_group = clib.invert(line_group)
        o_line_group = clib.offset(line_group, -d)
        o_line_group.insertFilet()
        o_line_group_list.append(o_line_group)
            
        i += 1
    
    if plotGraph == True:
        plt.figure(figsize=(10.0, 8.0))
        for line_group in line_group_list:
            for line in line_group.lines:
                plt.plot(line.x, line.y, "b")
        
        for o_line_group in o_line_group_list:
            init_line = o_line_group.lines[0]
            plt.plot(init_line.st[0], init_line.st[1], "bo")        
            for o_line in o_line_group.lines:
                plt.plot(o_line.x, o_line.y, "b--")    
                x0,y0,x1,y1 = getQuiver(o_line)
                plt.quiver(x0,y0,x1,y1, \
                   angles='xy',scale_units='xy',scale=1, width=0.003, color = 'black')  
        
        plt.axis("equal")
        plt.title("Example of 5.3.3")
        plt.savefig("res/Example of 5.3.3.svg")
        plt.show()


def example_5_4(plotGraph):
    doc = ez.readfile("test1.dxf")
    msp = doc.modelspace()
    lines = clib.importLinesFromDxf(msp, "LINE")
    splines = clib.importLinesFromDxf(msp, "SPLINE")

    all_lines = lines + splines
    line_group_list = clib.detectLineGroups(all_lines, 2)
    d = 2
    
    i = 0
    o_line_group_list = []
    for line_group in line_group_list:
        if line_group.ccw == False:
            line_group = clib.invert(line_group)
        o_line_group = clib.offset(line_group, -d)
        o_line_group.insertFilet()
        o_line_group_list.append(o_line_group)
        i += 1
    
    # Z軸の値を定義
    z_high = 60
    z_low = 0
    
    #送り速度の定義
    fs = 1000 # 早送り
    cs = 300 # 加工
    ms = 100 # ミリング速度
    
    # 加工前のマシン設定
    g_code_str = "T1\nG17 G49 G54 G80 G90 G94 G21 G40\n"
    
    i = 0
    while i < len(o_line_group_list):
        line_group = o_line_group_list[i]
        # 開始点へ移動
        g_code_str += clib.genGCodeStr([line_group.st[0]], [line_group.st[1]], [z_high], fs, "G0")
        # Z軸をミリング
        g_code_str += clib.genGCodeStr([line_group.st[0]], [line_group.st[1]], [z_low], ms, "G1")
        # 加工
        g_code_str += clib.genGCodeStr(line_group.x, line_group.y, [z_low]*len(line_group.x), cs, "G1")
        # Z方向に退避
        g_code_str += clib.genGCodeStr([line_group.ed[0]], [line_group.ed[1]], [z_high], ms, "G1")
        i += 1
        
    # Gコード出力用の文字列を確認
    print("G Code is bellow")
    print(g_code_str)
    
    # スクリプトファイル（.scr）として保存
    f = open("example_5_4.nc", "w")
    f.writelines(g_code_str)
    f.close()


def getQuiver(line):
    norm = clib.norm(line.x[0], line.y[0], line.x[1], line.y[1])
    x1 = (line.x[1]-line.x[0])/norm
    y1 = (line.y[1]-line.y[0])/norm
    
    return line.x[0], line.y[0], x1, y1

if __name__ == '__main__':
    meka_rib_demo(True)
    #example_3_2_1_1(True)
    #example_3_2_2_1(True)
    #example_3_5_1(True)
    #example_3_5_2(True)
    #example_3_5_3(True)
    #example_3_5_4(True)
    #example_3_6_1_1(True)
    #example_3_6_1_2(True)
    #example_3_6_1_3(True)
    #example_3_6_2(True)
    #example_3_7(True)
    #example_3_9_1(True)
    #example_3_9_2(True)
    #example_3_9_3(True)
    #example_4_1(True)
    #example_4_2(True)
    #example_4_3(True)
    #example_4_4(True)
    #example_4_6(True)
    #example_4_7(True)
    #example_4_8_1(True)
    #example_4_8_2(True)
    #example_4_9_2(True)
    #example_4_9_3(True)
    #example_4_9_4(True)
    #example_4_10(True)
    #example_4_11_1(True)
    #example_4_11_2(True)
    #example_4_11_3(True)
    #example_4_11_4(True)
    #example_4_12_1(True)
    #example_4_12_2(True)
    #example_5_1(True)
    #example_5_2_1(True)
    #example_5_2_2(True)
    #example_5_3_1_1(True)
    #example_5_3_1_2(True)
    #example_5_3_2(True)
    #example_5_3_3(True)
    #example_5_4(True)

    
    