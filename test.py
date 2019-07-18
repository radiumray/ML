
import numpy as np
import matplotlib.pyplot as plt
from LinearRegression.LinearModel import LinearRegressionUsingGD
from LinearRegression.Metrics import PerformanceMetrics
from LinearRegression.Plots import scatter_plot, plot, ploty


def generate_data_set():
        np.random.seed(0)
        x = np.random.rand(100, 1)
        y = 2 + 5 * x + np.random.rand(100, 1)
        return x, y


if __name__ == "__main__":
        # 初始化线性回归模型,设置学习率
        linear_regression_model = LinearRegressionUsingGD(0.5)

        # 生成训练数据
        x, y = generate_data_set()

        # linear_regression_model.fit(x, y)

        # 训练并显示过程
        linear_regression_model.fitShow(x, y)

        # 预测值
        predicted_values = linear_regression_model.predict(x)

        # 打印训练好的模型参数
        print(linear_regression_model.w_)
        intercept, coeffs = linear_regression_model.w_

        # 获得成本函数
        cost_function = linear_regression_model.cost_

        # 显示损失率的变化
        scatter_plot(x, y)
        plot(x, predicted_values)
        ploty(cost_function, 'no of iterations', 'cost function')

        # computing metrics
        metrics = PerformanceMetrics(y, predicted_values)
        rmse = metrics.compute_rmse()
        r2_score = metrics.compute_r2_score()

        print('The coefficient is {}'.format(coeffs))
        print('The intercept is {}'.format(intercept))
        print('Root mean squared error of the model is {}.'.format(rmse))
        print('R-squared score is {}.'.format(r2_score))
