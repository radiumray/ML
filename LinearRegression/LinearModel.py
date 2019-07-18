# imports
import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionUsingGD:
    """Linear Regression Using Gradient Descent.
    Parameters
    ----------
    eta : float
        Learning rate
    n_iterations : int
        No of passes over the training set
    Attributes
    ----------
    w_ : weights/ after fitting the model
    cost_ : total error of the model after each iteration
    """

    def __init__(self, eta=0.05, n_iterations=1000):
        self.eta = eta
        self.n_iterations = n_iterations



    def showCost(self, x, y):

        m = x.shape[0]
        x_train = np.c_[np.ones((m, 1)), x]

        self.cost_ = []
        self.w_ = np.zeros((x_train.shape[1], 1))

        y_pred = np.dot(x_train, self.w_)
        residuals = y_pred - y

        gradient_vector = np.dot(x_train.T, residuals)
        print(gradient_vector)


        # for _ in range(self.n_iterations):
        #     y_pred = np.dot(x_train, self.w_)
        #     residuals = y_pred - y
        #     cost = np.sum((residuals ** 2)) / (2 * m)
        #     self.cost_.append(cost)

        #     gradient_vector = np.dot(x_train.T, residuals)
        #     self.w_ -= (self.eta / m) * gradient_vector

        # return self







    def fit(self, x, y):
        """Fit the training data
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        Returns
        -------
        self : object
        """

        m = x.shape[0]
        x_train = np.c_[np.ones((m, 1)), x]

        self.cost_ = []
        self.w_ = np.zeros((x_train.shape[1], 1))

        # m = x.shape[0]

        for _ in range(self.n_iterations):
            y_pred = np.dot(x_train, self.w_)
            residuals = y_pred - y
            gradient_vector = np.dot(x_train.T, residuals)
            self.w_ -= (self.eta / m) * gradient_vector
            cost = np.sum((residuals ** 2)) / (2 * m)
            self.cost_.append(cost)
        return self


    def pltAnima(self, x, y):

        # 清除原有图像
        plt.cla()

        plt.xlabel('x')
        plt.ylabel('y')
        # 设定标题等
        plt.title('hhh')
        plt.grid(True)

        plt.scatter(x, y, c='r')

        predicted_values = self.predict(x)
        plt.plot(x, predicted_values, label='fit line')

        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        # plt.legend(loc="upper left", shadow=True)
        plt.legend()
        # 暂停
        plt.pause(0.001)


    def fitShow(self, x, y):
        """Fit the training data
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        Returns
        -------
        self : object
        """

        m = x.shape[0]
        x_train = np.c_[np.ones((m, 1)), x]

        self.cost_ = []
        self.w_ = np.zeros((x_train.shape[1], 1))
        # m = x.shape[0]

        for _ in range(self.n_iterations):

            # 通过当前权重显示拟合线
            self.pltAnima(x, y)

            # 通过当前权重得到数据点x相应的预测y值
            y_pred = np.dot(x_train, self.w_)
            # 预测的y值和实际的y值的差我们称之为残差
            residuals = y_pred - y
            # 残差的平方和就是成本函数
            cost = np.sum((residuals ** 2)) / (2 * m)

            # 计算所有参数的偏导数向量
            gradient_vector = np.dot(x_train.T, residuals)
            # 通过向量更新权重
            self.w_ -= (self.eta / m) * gradient_vector
            
            self.cost_.append(cost)
        
        # 关闭交互模式
        plt.ioff()

        # 图形显示
        plt.show()

        return self


    def predict(self, x):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        m = x.shape[0]
        x_train = np.c_[np.ones((m, 1)), x]
        return np.dot(x_train, self.w_)