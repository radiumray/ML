'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D

# 解决中文乱码问题
myfont = fm.FontProperties(fname="simsun.ttc", size=14)
matplotlib.rcParams["axes.unicode_minus"] = False


def simple_plot():
    """
    simple plot
    """
    # 生成画布
    plt.figure(figsize=(8, 6), dpi=80)

    # 打开交互模式
    plt.ion()

    # 循环
    for index in range(100):
        # 清除原有图像
        plt.cla()

        # 设定标题等
        plt.title("动态曲线图", fontproperties=myfont)
        plt.grid(True)

        # 生成测试数据
        x = np.linspace(-np.pi + 0.1*index, np.pi+0.1*index, 256, endpoint=True)
        y_cos, y_sin = np.cos(x), np.sin(x)

        # 设置X轴
        plt.xlabel("X轴", fontproperties=myfont)
        plt.xlim(-4 + 0.1*index, 4 + 0.1*index)
        plt.xticks(np.linspace(-4 + 0.1*index, 4+0.1*index, 9, endpoint=True))

        # 设置Y轴
        plt.ylabel("Y轴", fontproperties=myfont)
        plt.ylim(-1.0, 1.0)
        plt.yticks(np.linspace(-1, 1, 9, endpoint=True))

        # 画两条曲线
        plt.plot(x, y_cos, "b--", linewidth=2.0, label="cos示例")
        plt.plot(x, y_sin, "g-", linewidth=2.0, label="sin示例")

        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        plt.legend(loc="upper left", prop=myfont, shadow=True)

        # 暂停
        plt.pause(0.1)

    # 关闭交互模式
    plt.ioff()

    # 图形显示
    plt.show()
    return
# simple_plot()


def scatter_plot():
    """
    scatter plot
    """
    # 打开交互模式
    plt.ion()

    # 循环
    for index in range(50):
        # 清除原有图像
        # plt.cla()

        # 设定标题等
        plt.title("动态散点图", fontproperties=myfont)
        plt.grid(True)

        # 生成测试数据
        point_count = 5
        x_index = np.random.random(point_count)
        y_index = np.random.random(point_count)

        # 设置相关参数
        color_list = np.random.random(point_count)
        scale_list = np.random.random(point_count) * 100

        # 画散点图
        plt.scatter(x_index, y_index, s=scale_list, c=color_list, marker="o")

        # 暂停
        plt.pause(0.2)

    # 关闭交互模式
    plt.ioff()

    # 显示图形
    plt.show()
    return
# scatter_plot()

def three_dimension_scatter():
    """
    3d scatter plot
    """
    # 生成画布
    fig = plt.figure()

    # 打开交互模式
    plt.ion()

    # 循环
    for index in range(50):
        # 清除原有图像
        fig.clf()

        # 设定标题等
        fig.suptitle("三维动态散点图", fontproperties=myfont)

        # 生成测试数据
        point_count = 100
        x = np.random.random(point_count)
        y = np.random.random(point_count)
        z = np.random.random(point_count)
        color = np.random.random(point_count)
        scale = np.random.random(point_count) * 100

        # 生成画布
        ax = fig.add_subplot(111, projection="3d")

        # 画三维散点图
        ax.scatter(x, y, z, s=scale, c=color, marker=".")

        # 设置坐标轴图标
        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")

        # 设置坐标轴范围
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        # 暂停
        plt.pause(0.2)

    # 关闭交互模式
    plt.ioff()

    # 图形显示
    plt.show()
    return


three_dimension_scatter()
# simple_plot()

'''

'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style


# 构造数据
def get_data(sample_num=10000):
    """
    拟合函数为
    y = 5*x1 + 7*x2
    :return:
    """
    x1 = np.linspace(0, 9, sample_num)
    x2 = np.linspace(4, 13, sample_num)
    x = np.concatenate(([x1], [x2]), axis=0).T
    y = np.dot(x, np.array([5, 7]).T)
    return x, y
# 梯度下降法


def GD(samples, y, step_size=0.01, max_iter_count=1000):
    """
    :param samples: 样本
    :param y: 结果value
    :param step_size: 每一接迭代的步长
    :param max_iter_count: 最大的迭代次数
    :param batch_size: 随机选取的相对于总样本的大小
    :return:
    """
    # 确定样本数量以及变量的个数初始化theta值
    m, var = samples.shape
    theta = np.zeros(2)
    y = y.flatten()
    # 进入循环内
    print(samples)
    loss = 1
    iter_count = 0
    iter_list = []
    loss_list = []
    theta1 = []
    theta2 = []
    # 当损失精度大于0.01且迭代此时小于最大迭代次数时，进行
    while loss > 0.001 and iter_count < max_iter_count:
        loss = 0
        # 梯度计算
        theta1.append(theta[0])
        theta2.append(theta[1])
        for i in range(m):
            h = np.dot(theta, samples[i].T)
        # 更新theta的值,需要的参量有：步长，梯度
            for j in range(len(theta)):
                theta[j] = theta[j] - step_size*(1/m)*(h - y[i])*samples[i, j]
        # 计算总体的损失精度，等于各个样本损失精度之和
        for i in range(m):
            h = np.dot(theta.T, samples[i])
            # 每组样本点损失的精度
            every_loss = (1/(var*m))*np.power((h - y[i]), 2)
            loss = loss + every_loss

        print("iter_count: ", iter_count, "the loss:", loss)

        iter_list.append(iter_count)
        loss_list.append(loss)

        iter_count += 1
    plt.plot(iter_list, loss_list)
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.show()
    return theta1, theta2, theta, loss_list


def painter3D(theta1, theta2, loss):
    style.use('ggplot')
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    x, y, z = theta1, theta2, loss
    ax1.plot_wireframe(x, y, z, rstride=5, cstride=5)
    ax1.set_xlabel("theta1")
    ax1.set_ylabel("theta2")
    ax1.set_zlabel("loss")
    plt.show()


def predict(x, theta):
    y = np.dot(theta, x.T)
    return y


if __name__ == '__main__':
    samples, y = get_data()
    theta1, theta2, theta, loss_list = GD(samples, y)
    print(theta)  # 会很接近[5, 7]
    painter3D(theta1, theta2, loss_list)
    predict_y = predict(theta, [7, 8])
    print(predict_y)
'''


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
LR = 0.1
REAL_PARAMS = [1.2, 2.5]
INIT_PARAMS = [[5, 4],
               [5, 1],
               [2, 4.5]][2]


x = np.linspace(-1, 1, 200, dtype=np.float32)   # x data


def y_fun(a, b): return np.sin(b*np.cos(a*x))

def tf_y_fun(a, b): return tf.sin(b*tf.cos(a*x))


noise = np.random.randn(200)/10
y = y_fun(*REAL_PARAMS) + noise         # target

# tensorflow graph
a, b = [tf.Variable(initial_value=p, dtype=tf.float32) for p in INIT_PARAMS]
pred = tf_y_fun(a, b)
mse = tf.reduce_mean(tf.square(y-pred))
train_op = tf.train.GradientDescentOptimizer(LR).minimize(mse)

a_list, b_list, cost_list = [], [], []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t in range(400):
        a_, b_, mse_ = sess.run([a, b, mse])
        a_list.append(a_)
        b_list.append(b_)
        cost_list.append(mse_)    # record parameter changes
        # training
        result, _ = sess.run([pred, train_op])


# visualization codes:
print('a=', a_, 'b=', b_)
plt.figure(1)
plt.scatter(x, y, c='b')    # plot data
plt.plot(x, result, 'r-', lw=2)   # plot line fitting
# 3D cost figure
fig = plt.figure(2)
ax = Axes3D(fig)
a3D, b3D = np.meshgrid(np.linspace(-2, 7, 30),
                       np.linspace(-2, 7, 30))  # parameter space
cost3D = np.array([np.mean(np.square(y_fun(a_, b_) - y))
                   for a_, b_ in zip(a3D.flatten(), b3D.flatten())]).reshape(a3D.shape)
ax.plot_surface(a3D, b3D, cost3D, rstride=1, cstride=1,
                cmap=plt.get_cmap('rainbow'), alpha=0.5)
ax.scatter(a_list[0], b_list[0], zs=cost_list[0],
           s=300, c='r')  # initial parameter place
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.plot(a_list, b_list, zs=cost_list, zdir='z',
        c='r', lw=3)    # plot 3D gradient descent
plt.show()
