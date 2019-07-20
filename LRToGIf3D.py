
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import misc
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

numRow=200

learning_rate=0.02
precision = 0.0001 #当学习进步幅度小于此值时退出学习算法
previous_step_size = 1 # 上一次学习的进步值
max_iters = 200 # 最大迭代次数
iters = 0 #当前迭代数

# 解决中文乱码问题
myfont = fm.FontProperties(fname="simsun.ttc", size=14)

zoomScale=2.0

interceptArray=np.linspace(-zoomScale, zoomScale, 30)
slopeArray=np.linspace(-zoomScale, zoomScale, 30)

REAL_PARAMS = [-1, 0.5]

# 生成测试数据
x = np.linspace(-zoomScale, zoomScale, numRow, dtype=np.float32)

xLen=x.shape[0]
x=x.reshape((xLen,1))

x_b=np.c_[np.ones((numRow, 1)), x]

def custFunction(intercept, slope):
    # return np.sin(intercept * np.cos(slope * x))
    return slope * x + intercept

# 得到函数y的所有值
noise = np.random.randn(numRow)/10

noiseLen=noise.shape[0]
noise=noise.reshape((noiseLen,1))

y = custFunction(*REAL_PARAMS) + noise

theta_List=[]

interceptList=[]
slopeList=[]
costList = []

ModeDebug=True
# ModeDebug=False

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
        plt.xlim(-zoomScale, zoomScale)
        plt.ylim(-zoomScale, zoomScale)

        # # 画曲线
        # plt.scatter(x, y, c='r', alpha=0.5, label="散点数据")
        # # 从列表里取出相应x值
        # intercept, slope =theta_List[index]
        # print('inter', intercept)
        # print('slope', slope)
        # yLine = custFunction(intercept, slope)
        # plt.plot(x, yLine, '-g', label='拟合线')



        # ax = fig.gca(projection='3d')
        ax = Axes3D(fig)
        ax.plot_surface(intercept3D, slope3D, cost3D, rstride=1, cstride=1,
                        cmap=plt.get_cmap('rainbow'), alpha=0.5)
        

        ax.scatter(interceptList[index], slopeList[index], costList[index],
                s=30, c='g')  # initial parameter place
        ax.set_xlabel('截距', fontproperties=myfont)
        ax.set_ylabel('斜率', fontproperties=myfont)
        ax.set_zlabel('成本', fontproperties=myfont)
        print(costList[index])
        ax.plot(interceptList[:index], slopeList[:index], zs=costList[:index], zdir='z',
        c='g', lw=1, linestyle='--')


        plt.legend(loc="lower right", prop=myfont, shadow=True)
        # 暂停
        plt.pause(0.001)


# 生成画布
fig = plt.figure(figsize=(8, 6), dpi=80)

# theta=np.random.randn(2,1)
theta=np.array([[2], [2]])

while(iters < max_iters):


    y_pred=np.dot(x_b, theta)
    residuals=y_pred-y
    cost=np.sum(np.square(residuals))/2*numRow
    # cost=np.mean(np.square(residuals))
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
# cost3D = np.array([np.mean(np.square(custFunction(intercept, slope) - y)) for intercept, slope in zip(intercept3D.flatten(), slope3D.flatten())]).reshape(intercept3D.shape)
cost3D = np.array([np.sum(np.square(custFunction(intercept, slope) - y))/2*numRow for intercept, slope in zip(intercept3D.flatten(), slope3D.flatten())]).reshape(intercept3D.shape)

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


