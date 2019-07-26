import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def nomalize(X):
    mean = np.mean(X)
    std=np.std(X)
    X=(X-mean)/std
    return X

def apend_bias_reshape(features, labels):
    m=features.shape[0]
    n=features.shape[1]
    x=np.reshape(np.c_[np.ones(m), features], [m,n+1])
    y=np.reshape(labels, [m,1])
    return x,y

# # Data
# tf.data.
# boston = tf.contrib.learn.datasets.load_dataset('boston')
# X_train, Y_train=boston.data, boston.target

#波士顿房价数据
boston=tf.keras.datasets.boston_housing.load_data()
(X_train, Y_train), (X_test, Y_test) = boston
X_train = nomalize(X_train)
X_train, Y_train = apend_bias_reshape(X_train, Y_train)
m, n = X_train.shape


# 为训练数据声明TensorFlow占位符。观测占位符X的形状变化
# Placeholder for the Training Data
X = tf.placeholder(tf.float32, name='X', shape=[m, n])
Y = tf.placeholder(tf.float32, name='Y')

# 为权重和偏置创建TensorFlow变量。通过随机数初始化权重：
# Variables for coefficients
w = tf.Variable(tf.random_normal([n, 1]))

# 定义要用于预测的线性回归模型。现在需要矩阵乘法来完成这个任务
# The Linear Regression Model
Y_hat = tf.matmul(X, w)

# 为了更好地求微分，定义损失函数：
# Loss function
loss=tf.reduce_mean(tf.square(Y-Y_hat, name='loss'))

# 选择正确的优化器
# Gradient Descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# 定义初始化操作符
# Initializing Variables
init_op = tf.global_variables_initializer()
total = []

# 开始计算图
with tf.Session() as sess:
    # Initialize variables
    sess.run(init_op)
    writer = tf.summary.FileWriter('graphs', sess.graph)
    # train the model for 100 epcohs
    for i in range(100):
        _, l = sess.run([optimizer, loss], feed_dict={X:X_train, Y:Y_train})
        total.append(l)
        print('Epoch {0}: Loss {1}'.format(i, l))
    writer.close()
    # w_value, b_value = sess.run([w, b])

plt.plot(total)
plt.show()





