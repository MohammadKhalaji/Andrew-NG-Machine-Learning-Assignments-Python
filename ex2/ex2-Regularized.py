import numpy as np
import matplotlib.pyplot as plot
from scipy.optimize import *


def main():

    file_name = 'ex2data2.txt'
    X, y, m, n = read_file(file_name, ',')
    originalX = X
    X, n = map_features(X)
    theta = np.zeros((n, 1))


    # overfit
    ans1 = minimize(lambda t: cost(X, y, t, lmd=0), theta, method='Nelder-Mead').x.reshape((n, 1))
    print('Overfit training accuracy: %{0}'.format(get_accuracy(X, y, ans1)))


    # fine
    ans2 = minimize(lambda t: cost(X, y, t, lmd=2), theta, method='Nelder-Mead').x.reshape((n, 1))
    print('Fine training accuracy: %{0}'.format(get_accuracy(X, y, ans2)))

    #underfit
    ans3 = minimize(lambda t: cost(X, y, t, lmd=100), theta, method='Nelder-Mead').x.reshape((n, 1))
    print('Underfit training accuracy: %{0}'.format(get_accuracy(X, y, ans3)))

    plot_decision_boundary(X, ans2, originalX)
    plot_points(originalX, y)

    plot.xlabel('Microchip Test 1')
    plot.ylabel('Microchip Test 2')
    plot.title('Logistic Regression - Regularized\nPolynomial Features  ')
    plot.legend()
    plot.savefig('ex2-Regularized.png', dpi=300)
    plot.show()



    while True:
        inp = input('Enter x to predict value, split by comma:\n')
        inp = inp.replace(' ', '')
        inp = inp.split(',')
        inp = list(map(float, inp))
        x = np.array([inp]).T
        print('resutl: ', predict(x, ans2))
        print('----------------------')


def read_file(file_name, delimiter):
    points = np.loadtxt(file_name, delimiter=delimiter)
    # x0s = np.ones((points.shape[0], 1))
    # points = np.append(x0s, points, axis=1)
    m = points.shape[0]
    n = points.shape[1] - 1
    return points[..., 0:n], points[..., n:], m, n


def plot_points(X, y):
    z_x = []
    z_y = []
    o_x = []
    o_y = []
    for i in range(0, X.shape[0]):
        if y[i][0] == 1:
            o_x.append(X[i][0])
            o_y.append(X[i][1])
        else:
            z_x.append(X[i][0])
            z_y.append(X[i][1])
    plot.scatter(o_x, o_y, marker='+', c='g', label='y = 1')
    plot.scatter(z_x, z_y, marker='.', c='r', label='y = 0')


def map_features(X):
    deg = 7
    k = 0
    newX = []
    cur = []
    m = X.shape[0]
    for k in range(0, m):
        newX.append(map_single_feature(X[k]))
    X = np.array(newX)
    return X, X.shape[1]


def map_single_feature(x):
    deg = 7
    cur = []
    for i in range(0, deg):
        for j in range(0, i+1):
            cur.append(np.power(x[0], i-j) * np.power(x[1], j))
    return cur


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def h(X, theta):
    return sigmoid(np.matmul(X, theta))


def single_h(x, theta):
    return sigmoid(np.matmul(theta.T, x))


def predict(x, theta):
    if single_h(x, theta).reshape((1, 1))[0][0] >= 0.5:
        return 1
    return 0


def cost(X, y, theta, lmd):
    n = X.shape[1]
    m = X.shape[0]
    theta = theta.reshape((n, 1))
    # we do not penalize theta0
    penalize = np.matmul(theta.T, theta) - theta[0][0]**2
    ans = np.matmul(y.T, np.log(h(X, theta))) + np.matmul((1 - y).T, np.log(1 - h(X, theta)))
    ans = ans * (-1 / m)
    ans += (lmd / (2 * m)) * penalize
    return ans


def plot_decision_boundary(X, theta, originalX):
    # Build the grid to plot:
    x_min, x_max = originalX[..., 0].min() - 0.25, originalX[..., 0].max() + 0.25
    y_min, y_max = originalX[..., 1].min() - 0.25, originalX[..., 1].max() + 0.25
    x = np.arange(x_min, x_max, 0.01)
    y = np.arange(y_min, y_max, 0.01)
    xx, yy = np.meshgrid(x, y)

    # Build the data:
    zz = np.zeros(xx.shape)
    for i in range(0, xx.shape[0]):
        for j in range(0, xx.shape[1]):
            my_x = [xx[i][j], yy[i][j]]
            my_x = map_single_feature(my_x)
            zz[i][j] = predict(my_x, theta)

    plot.pcolormesh(xx, yy, zz)


def get_accuracy(X, y, theta):
    m = X.shape[0]
    corrects = 0
    for i in range(0, m):
        x = np.array([X[i]]).T
        if predict(x, theta) == y[i]:
            corrects += 1
    return 100 * corrects / m


if __name__ == '__main__':
    main()