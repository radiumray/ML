import numpy as np
import matplotlib.pyplot as plt
from LinearRegression.LinearModel import LinearRegressionUsingGD

def generate_data_set():
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 2 + 5 * x + np.random.rand(100, 1)
    return x, y


# 初始化线性回归模型,设置学习率
linear_regression_model = LinearRegressionUsingGD(0.5)
# 生成训练数据
x, y = generate_data_set()

linear_regression_model.showCost(x, y)





# 训练并显示过程
# linear_regression_model.fitShow(x, y)


