

'''
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
max_iters = 10000 # 最大迭代次数
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

ModeDebug=True
# ModeDebug=False

def animate(index):

    if(index%1==0): # 用来压缩gif
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


'''

'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

n_radii = 8
n_angles = 36

# Make radii and angles spaces (radius r=0 omitted to eliminate duplication).
radii = np.linspace(0.125, 1.0, n_radii)
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

# Repeat all angles for each radius.
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

# Convert polar (radii, angles) coords to cartesian (x, y) coords.
# (0, 0) is manually added at this stage,  so there will be no duplicate
# points in the (x, y) plane.

x = np.append(0, (radii*np.cos(angles)).flatten())

y = np.append(0, (radii*np.sin(angles)).flatten())

# Compute z to make the pringle surface.
z = np.sin(-x*y)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

plt.show()
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


learning_rate=0.02
# precision = 0.000001 #当学习进步幅度小于此值时退出学习算法
precision = 0.0001 #当学习进步幅度小于此值时退出学习算法
previous_step_size = 1 # 上一次学习的进步值
# max_iters = 10000 # 最大迭代次数
max_iters = 200 # 最大迭代次数
iters = 0 #当前迭代数
numRow=200

theta_List=[]
interceptList=[]
slopeList=[]
costList = []

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.

X = np.random.uniform(-10, 10, (numRow, 1))
def custFunction(x):
    return -1+0.5*x+np.random.randn(numRow, 1)
# 得到函数y的所有值
Y = custFunction(X)

X_b=np.c_[np.ones((numRow, 1)), X]

theta=np.random.randn(2,1)

while(iters < max_iters):
    y_pred=np.dot(X_b, theta)
    residuals=y_pred-Y
    cost=np.sum(residuals**2)/2*numRow

    costList.append(cost)
    theta_List.append(theta)
    # interceptList.append(theta[0])
    # slopeList.append(theta[1])

    # 2.求梯度gradinet
    gradients = 1/numRow *X_b.T.dot(X_b.dot(theta)-Y)
    # 3.用公式调整theta值，theta_t+1 = theta_t - grad * learning_rate
    theta = theta - learning_rate * gradients
    iters = iters+1 #更新迭代次数
    
lenTheta_List=len(theta_List)

for index in range(lenTheta_List):
    intercept, slope = theta_List[index]
    interceptList.append(intercept[0])
    slopeList.append(slope[0])

interceptList=np.array(interceptList)
slopeList=np.array(slopeList)
costList=np.array(costList)

# X, Y = np.meshgrid(X, Y)
# Z = X


# interceptAix, slopeAix = np.meshgrid(interceptList, slopeList)
# costAix = costList

ax.plot(interceptList, slopeList, costList, label='gradient')

# ax.plot_trisurf(interceptList, slopeList, costList, linewidth=0.2, antialiased=True)


ax.set_title("gradient")
ax.set_xlabel("intercept")
ax.set_ylabel("slope")
ax.set_zlabel("cost")



plt.show()

'''
# Plot the surface.
surf = ax.plot_surface(interceptAix, slopeAix, costAix, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
'''

'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

numRow=200


# x = np.random.uniform(-10, 10, (numRow, 1))

# x_b=np.c_[np.ones((numRow, 1)), x]

# def custFunction(x):
#     return -1+0.5*x+np.random.randn(numRow, 1)

# # 得到函数y的所有值
# y = custFunction(x)

fig = plt.figure(figsize=(12, 8))
ax = Axes3D(fig)
delta = 0.125
# 生成代表X轴数据的列表
# x = np.arange(-3.0, 3.0, delta)

x = np.random.uniform(-1, 1, (numRow, 1))

def custFunction(x):
    return -1+0.5*x+np.random.randn(numRow, 1)

# 生成代表Y轴数据的列表
# y = np.arange(-2.0, 2.0, delta)

# 得到函数y的所有值
y = custFunction(x)

# 对x、y数据执行网格化
X, Y = np.meshgrid(x, y)


# Z1 = np.exp(-X**2 - Y**2)
# Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
# # 计算Z轴数据（高度数据）
# Z = (Z1 - Z2) * 2

Z = X*X

# 绘制3D图形
ax.plot_surface(X, Y, Z,
                rstride=1,  # rstride（row）指定行的跨度
                cstride=1,  # cstride(column)指定列的跨度
                cmap=plt.get_cmap('rainbow'))  # 设置颜色映射
# 设置Z轴范围
ax.set_zlim(-2, 2)
# 设置标题
plt.title("3D图")
plt.show()
'''