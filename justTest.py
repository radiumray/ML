'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


learning_rate=0.02
# precision = 0.000001 #当学习进步幅度小于此值时退出学习算法
precision = 0.0001 #当学习进步幅度小于此值时退出学习算法
previous_step_size = 1 # 上一次学习的进步值
# max_iters = 10000 # 最大迭代次数
max_iters = 200 # 最大迭代次数
iters = 0 #当前迭代数

REAL_PARAMS = [-1, 0.5]

numRow=200


x = np.linspace(-10, 10, numRow, dtype=np.float32)   # x data

def custFunction(intercept, slope):
    return slope*x + intercept

noise = np.random.randn(numRow)/10
y = custFunction(*REAL_PARAMS) + noise         # target


interceptArray=np.linspace(-10, 10, 30)
slopeArray=np.linspace(-10, 10, 30)

theta_List=[]
interceptList=[]
slopeList=[]
costList = []


X_b=np.c_[np.ones((numRow, 1)), x]

theta=np.random.randn(2,1)

while(iters < max_iters):
    y_pred=np.dot(X_b, theta)
    residuals=y_pred-y
    cost=np.sum(residuals**2)/2*numRow
    costList.append(cost)
    

    # 2.求梯度gradinet
    gradients = 1/numRow *X_b.T.dot(X_b.dot(theta)-y)
    # 3.用公式调整theta值，theta_t+1 = theta_t - grad * learning_rate
    theta = theta - learning_rate * gradients
    iters = iters+1 #更新迭代次数
    theta_List.append(theta)

lenTheta_List=len(theta_List)


for index in range(lenTheta_List):
    intercept, slope = theta_List[index]
    interceptList.append(intercept[0])
    slopeList.append(slope[0])

interceptList=np.array(interceptList)
slopeList=np.array(slopeList)
costList=np.array(costList)


intercept3D, slope3D = np.meshgrid(interceptArray, slopeArray)  # parameter space
cost3D = np.array([np.mean(np.square(custFunction(intercept, slope) - y)) for intercept, slope in zip(intercept3D.flatten(), slope3D.flatten())]).reshape(intercept3D.shape)




plt.figure(1)
plt.scatter(x, y, c='b')    # plot data

print('best intercept', interceptList[lenTheta_List-1])
print('best slope', slopeList[lenTheta_List-1])


result=custFunction(interceptList[lenTheta_List-1], slopeList[lenTheta_List-1])

plt.plot(x, result, 'r-', lw=2)   
'''

'''
fig = plt.figure(2)
ax = fig.gca(projection='3d')
# ax = Axes3D(fig)


ax.plot_surface(intercept3D, slope3D, cost3D, rstride=1, cstride=1,
                cmap=plt.get_cmap('rainbow'), alpha=0.5)
ax.scatter(interceptList[0], slopeList[0], zs=costList[0],
           s=300, c='r')  # initial parameter place


ax.set_xlabel('intrcept')
ax.set_ylabel('slope')
ax.set_zlabel("cost")


# print(interceptList)

ax.plot(interceptList, slopeList, zs=costList, zdir='z',
        c='r', lw=3)    # plot 3D gradient descent

plt.show()
'''





'''
import numpy as np

__author__='ray'

numRow=100

X=2* np.random.rand(numRow, 1)
y=4+3*X+np.random.randn(numRow, 1)

X_b=np.c_[np.ones((numRow, 1)), X]

learning_rate=0.1
n_iteration=1000


# 1.初始化theta, w0,w1..............wn
theta=np.random.randn(2,1)
count=0

for iteration in range(n_iteration):
    count+=1
    # 2.求梯度gradinet
    gradients = 1/numRow *X_b.T.dot(X_b.dot(theta)-y)
    # 3.用公式调整theta值，theta_t+1 = theta_t - grad * learning_rate
    theta = theta - learning_rate * gradients

print(count)
print(theta)
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


zoomScale=5.0

interceptArray=np.linspace(-zoomScale, zoomScale, 30)
slopeArray=np.linspace(-zoomScale, zoomScale, 30)



# 生成测试数据
xLine = np.arange(-zoomScale, zoomScale, 0.1)

# x=5*np.random.rand(numRow, 1)
x = np.random.uniform(-zoomScale, zoomScale, (numRow, 1))

x_b=np.c_[np.ones((numRow, 1)), x]

def custFunction(x):
    return -1+0.5*x+np.random.randn(numRow, 1)

def custFunction2(intercept, slope):
    return slope*x + intercept

# 得到函数y的所有值
y = custFunction(x)

theta_List=[]


interceptList=[]
slopeList=[]
costList = []

ModeDebug=True
# ModeDebug=False

def animate(index):

    if(index%1==0): # 用来压缩gif
        # 清除原有图像
        plt.cla()
        # 设定标题等
        plt.title("梯度下降", fontproperties=myfont)
        plt.grid(True)



        # # 设置X轴
        # plt.xlabel("X轴", fontproperties=myfont)
        # # 设置Y轴
        # plt.ylabel("Y轴", fontproperties=myfont)
        # plt.xlim(-zoomScale, zoomScale)
        # plt.ylim(-zoomScale, zoomScale)
        # # 画曲线
        # plt.scatter(x, y, c='r', alpha=0.5, label="散点数据")

        # # 从列表里取出相应x值
        # intercept, slope =theta_List[index]
        # yLine = slope*xLine+intercept
        # plt.plot(xLine, yLine, '-g', label='拟合线')



        # ax = fig.gca(projection='3d')
        ax = Axes3D(fig)

        ax.plot_surface(intercept3D, slope3D, cost3D, rstride=1, cstride=1,
                        cmap=plt.get_cmap('rainbow'), alpha=0.5)
        ax.scatter(interceptList[index], slopeList[index], costList[index],
                s=30, c='r')  # initial parameter place
        ax.set_xlabel('intrcept')
        ax.set_ylabel('slope')
        ax.set_zlabel("cost")

        print(costList[index])

        # intercept, slope =theta_List[index]
        # print('intercept', intercept)
        # print('slope', slope)
        ax.plot(interceptList, slopeList, zs=costList, zdir='z',
        c='r', lw=3)    # plot 3D gradient descent

        plt.legend(loc="lower right", prop=myfont, shadow=True)
        # 暂停
        plt.pause(0.001)


# 生成画布
fig = plt.figure(figsize=(8, 6), dpi=80)

theta=np.random.randn(2,1)

while(iters < max_iters):


    y_pred=np.dot(x_b, theta)
    residuals=y_pred-y
    # cost=np.sum(np.square(residuals))/2*numRow
    cost=np.mean(np.square(residuals - y))
    costList.append(cost)

    # 2.求梯度gradinet
    gradients = 1/numRow *x_b.T.dot(x_b.dot(theta)-y)
    # 3.用公式调整theta值，theta_t+1 = theta_t - grad * learning_rate
    theta = theta - learning_rate * gradients

    iters = iters+1 #更新迭代次数
    # print("Iteration",iters,"\nX value is",cur_x) #打印值
    theta_List.append(theta)

lenTheta_List=len(theta_List)

for index in range(lenTheta_List):
    intercept, slope = theta_List[index]
    interceptList.append(intercept[0])
    slopeList.append(slope[0])

interceptList=np.array(interceptList)
slopeList=np.array(slopeList)
costList=np.array(costList)

intercept3D, slope3D = np.meshgrid(interceptArray, slopeArray)  # parameter space
cost3D = np.array([np.mean(np.square(custFunction2(intercept, slope) - y)) for intercept, slope in zip(intercept3D.flatten(), slope3D.flatten())]).reshape(intercept3D.shape)



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


