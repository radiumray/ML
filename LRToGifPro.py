
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import misc
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

numRow=100

learning_rate=0.02
# precision = 0.000001 #当学习进步幅度小于此值时退出学习算法
precision = 0.0001 #当学习进步幅度小于此值时退出学习算法
previous_step_size = 1 # 上一次学习的进步值
max_iters = 100 # 最大迭代次数
iters = 0 #当前迭代数


# 解决中文乱码问题
myfont = fm.FontProperties(fname="simsun.ttc", size=14)

# 生成测试数据
xLine = np.arange(-10.0, 10.0, 0.1)

# x=5*np.random.rand(numRow, 1)
x = np.random.uniform(-10, 10, (numRow, 1))

x_b=np.c_[np.ones((numRow, 1)), x]

def custFunction(x):
    return -1+0.5*x+np.random.randn(numRow, 1)

# 得到函数y的所有值
y = custFunction(x)

theta_List=[]

# ModeDebug=True
ModeDebug=False

def animate(index):

    if(index%20==0): # 用来压缩gif
        # 清除原有图像
        plt.cla()
        # 设定标题等
        plt.title("梯度下降", fontproperties=myfont)
        plt.grid(True)
        # 设置X轴
        plt.xlabel("X轴", fontproperties=myfont)
        # 设置Y轴
        plt.ylabel("Y轴", fontproperties=myfont)
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        # 画曲线
        plt.scatter(x, y, c='r', alpha=0.5, label="散点数据")

        # 从列表里取出相应x值
        intercept, slope =theta_List[index]
        yLine = slope*xLine+intercept
        plt.plot(xLine, yLine, '-g', label='拟合线')

        # # 通过x得到抛物线上的y值
        # indexY=custFunction(indexX)
        # # 把这个点用绿颜色画出来
        # plt.scatter(indexX, indexY, s=30, color='g',  label='下降点')
        # # GDPLable = "$f(t)=e^{-t} \cos (2 \pi t)$"
        # # 标注这个点目前的值
        # plt.annotate(str(indexX),xy=(indexX,indexY),xytext=(indexX+1,indexY-1),arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))

        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        plt.legend(loc="lower right", prop=myfont, shadow=True)
        # 暂停
        plt.pause(0.001)


# 生成画布
fig = plt.figure(figsize=(8, 6), dpi=80)

theta=np.random.randn(2,1)

while(iters < max_iters):

    # 2.求梯度gradinet
    gradients = 1/numRow *x_b.T.dot(x_b.dot(theta)-y)
    # 3.用公式调整theta值，theta_t+1 = theta_t - grad * learning_rate
    theta = theta - learning_rate * gradients

    iters = iters+1 #更新迭代次数
    # print("Iteration",iters,"\nX value is",cur_x) #打印值
    theta_List.append(theta)

lenTheta_List=len(theta_List)

if(ModeDebug):
    # 打开交互模式
    plt.ion()

    for index in range(lenTheta_List):
        animate(index)

    # 关闭交互模式
    plt.ioff()
    # 图形显示
    plt.show()
else:
    anim = animation.FuncAnimation(fig, animate, frames=lenTheta_List)
    anim.save('hhh.gif', writer='imagemagick', fps=30)


