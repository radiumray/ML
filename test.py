
'''
import numpy as np
import matplotlib.pyplot as plt
from LinearRegression.LinearModel import LinearRegressionUsingGD
from LinearRegression.Metrics import PerformanceMetrics
from LinearRegression.Plots import scatter_plot, plot, ploty


def generate_data_set():
    """ Generates Random Data
    Returns
    -------
    x : array-like, shape = [n_samples, n_features]
            Training samples
    y : array-like, shape = [n_samples, n_target_values]
            Target values
    """
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 2 + 5 * x + np.random.rand(100, 1)
    return x, y


if __name__ == "__main__":
    # initializing the model
    linear_regression_model = LinearRegressionUsingGD(0.5)

    # generate the data set
    x, y = generate_data_set()

    # linear_regression_model.fit(x, y)

    linear_regression_model.fitShow(x, y)

    # predict values
    predicted_values = linear_regression_model.predict(x)

    # model parameters
    print(linear_regression_model.w_)
    intercept, coeffs = linear_regression_model.w_

    # cost_function
    cost_function = linear_regression_model.cost_

    # plotting
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

'''

for index in range(100):
        if(index%3==0):
                print(index)