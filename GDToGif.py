import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import misc
from matplotlib import animation

cur_x = 3 # 初始值x设置为3
# rate = 0.01 # 学习率
rate = 0.1 # 学习率
# precision = 0.000001 #当学习进步幅度小于此值时退出学习算法
precision = 0.0001 #当学习进步幅度小于此值时退出学习算法
previous_step_size = 1 # 上一次学习的进步值
max_iters = 10000 # 最大迭代次数
iters = 0 #当前迭代数

# 解决中文乱码问题
myfont = fm.FontProperties(fname="simsun.ttc", size=14)

# 生成测试数据
x = np.arange(-10.0, 10.0, 0.1)
numPoint = len(x)

curX_List=[]

# ModeDebug=True
ModeDebug=False

def animate(index):

    if(index%2==0): # 用来压缩gif
        # 清除原有图像
        plt.cla()
        # 设定标题等
        plt.title("梯度下降", fontproperties=myfont)
        plt.grid(True)
        # 得到函数y的所有值
        y = custFunction(x)
        # 设置X轴
        plt.xlabel("X轴", fontproperties=myfont)
        # 设置Y轴
        plt.ylabel("Y轴", fontproperties=myfont)
        # plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        # 画曲线
        plt.plot(x, y, 'r-', linewidth=2.0, alpha=0.5, label="函数")
        # 从列表里取出相应x值
        indexX=curX_List[index]
        # 通过x得到抛物线上的y值
        indexY=custFunction(indexX)
        # 把这个点用绿颜色画出来
        plt.scatter(indexX, indexY, s=30, color='g',  label='下降点')

        # GDPLable = "$f(t)=e^{-t} \cos (2 \pi t)$"
        # 标注这个点目前的值
        plt.annotate(str(indexX),xy=(indexX,indexY),xytext=(indexX+1,indexY-1),arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))

        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        plt.legend(loc="lower right", prop=myfont, shadow=True)
        # 暂停
        plt.pause(0.001)


def custFunction(x):
    return np.sin(x)
    # return 0.3*x*x+2*x+1
    # return x**2
    # return (x+5)**2


# 生成画布
fig = plt.figure(figsize=(8, 6), dpi=80)

while(previous_step_size > precision and iters < max_iters):

    curX_List.append(cur_x)
    prev_x = cur_x #更新上一次的值
    cur_x = cur_x - rate * misc.derivative(custFunction, prev_x) #梯度下降更新值
    previous_step_size = abs(cur_x - prev_x) #算出学习进步值
    iters = iters+1 #更新迭代次数
    # print("Iteration",iters,"\nX value is",cur_x) #打印值


lenCurXList=len(curX_List)


if(ModeDebug):
    # 打开交互模式
    plt.ion()

    for index in range(lenCurXList):
        animate(index)

    # 关闭交互模式
    plt.ioff()
    # 图形显示
    plt.show()
else:
    anim = animation.FuncAnimation(fig, animate, frames=lenCurXList)
    anim.save('hhh.gif', writer='imagemagick', fps=4)





# http://people.duke.edu/~ccc14/sta-663-2018/notebooks/S09G_Gradient_Descent_Optimization.html


'''
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2

def grad(x):
    return 2*x

def gd(x, grad, alpha, max_iter=10):
    xs = np.zeros(1 + max_iter)
    xs[0] = x
    for i in range(max_iter):
        x = x - alpha * grad(x)
        xs[i+1] = x
    return xs

def gd_momentum(x, grad, alpha, beta=0.9, max_iter=10):
    xs = np.zeros(1 + max_iter)
    xs[0] = x
    v = 0
    for i in range(max_iter):
        v = beta*v + (1-beta)*grad(x)
        vc = v/(1+beta**(i+1))
        x = x - alpha * vc
        xs[i+1] = x
    return xs


alpha = 0.1
x0 = 1
xs = gd(x0, grad, alpha)
xp = np.linspace(-1.2, 1.2, 100)
plt.plot(xp, f(xp))
plt.plot(xs, f(xs), 'o-', c='red')
for i, (x, y) in enumerate(zip(xs, f(xs)), 1):
    plt.text(x, y+0.2, i,
             bbox=dict(facecolor='yellow', alpha=0.5), fontsize=14)
plt.show()
'''
