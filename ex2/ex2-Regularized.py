import numpy as np
import matplotlib.pyplot as plot
from scipy.optimize import *


def main():
    file_name = 'ex2data2.txt'
    X, y, m, n = read_file(file_name, ',')
    plot_points(X, y)
    X, n = map_features(X)
    theta = np.zeros((n, 1))


    # overfit
    ans1 = minimize(lambda t: cost(X, y, t, lmd=0), theta, method='Nelder-Mead').x.reshape((n, 1))

    # fine
    ans2 = minimize(lambda t: cost(X, y, t, lmd=1), theta, method='Nelder-Mead').x.reshape((n, 1))

    #underfit
    ans3 = minimize(lambda t: cost(X, y, t, lmd=100), theta, method='Nelder-Mead').x.reshape((n, 1))

    print(h(np.array([0, 0, 0.2]).T, ans2))



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
    plot.scatter(o_x, o_y, marker='+', c='g')
    plot.scatter(z_x, z_y, marker='.', c='r')



def map_features(X):
    deg = 7
    k = 0
    newX = []
    cur = []
    m = X.shape[0]
    for k in range(0, m):
        newX.append(map_feature(X[k]))
    X = np.array(newX)
    return X, X.shape[1]


def map_feature(x):
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



if __name__ == '__main__':
    main()