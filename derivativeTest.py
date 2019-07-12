

'''
from pylab import *
from scipy import misc
ax = subplot(111)
def custFunction(x):
    return 3*x*x+2*x+1
x = arange(-2.0, 2.0, 0.01)
y = custFunction(x)
plot(x, y,'r-')
yp = misc.derivative(custFunction, x)
plot(x, yp,'b-')
grid(True)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
text(-0.75, 6.0,
     r'$f(x)=3x^2+2x+1$', horizontalalignment='center',
     fontsize=18,color='red')
text(-1.0, -8.0,
     r"$f'(x)=6x+2$", horizontalalignment='center',
     fontsize=18,color='blue')
show()
'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import misc 
# 解决中文乱码问题
myfont = fm.FontProperties(fname="simsun.ttc", size=14)

def custFunction(x):
    return np.sin(x)
    # return 3*x*x+2*x+1
    # return x*x

def simple_plot():
    """
    simple plot
    """
    # 生成画布
    plt.figure(figsize=(8, 6), dpi=80)

    # 打开交互模式
    plt.ion()

    # 生成测试数据
    x = np.arange(-10.0, 10.0, 0.1)
    numPoint=len(x)

    # 循环
    for index in range(numPoint):
        # 清除原有图像
        plt.cla()

        # 设定标题等
        plt.title("函数求导", fontproperties=myfont)
        plt.grid(True)

        # 得到函数y的所有值
        y = custFunction(x)
        # 得到函数导数y的所有值
        dy=misc.derivative(custFunction, x)

        # 设置X轴
        plt.xlabel("X轴", fontproperties=myfont)

        # 设置Y轴
        plt.ylabel("Y轴", fontproperties=myfont)
        # plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        # plt.yticks(np.linspace(-1, 1, 9, endpoint=True))

        # 画曲线
        plt.plot(x, y,'r-', linewidth=2.0, label="函数")
        # 画曲线导数
        plt.plot(x, dy, "g-", linewidth=2.0, label="函数的导数")

        # 得到给定x在函数上的y值
        yOfLine = custFunction(x[index])
        # 通过y值和导数方程得到此点在切线上的y值
        y_tan = misc.derivative(custFunction, x[index]) * (x - x[index]) + yOfLine 
        # 画切线
        plt.plot(x,y_tan,'b--', linewidth=2.0, label="函数的切线")




        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        plt.legend(loc="upper left", prop=myfont, shadow=True)

        # 暂停
        plt.pause(0.001)

    # 关闭交互模式
    plt.ioff()

    # 图形显示
    plt.show()
    return



simple_plot()
