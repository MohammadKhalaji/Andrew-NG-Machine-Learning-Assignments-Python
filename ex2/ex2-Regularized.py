import numpy as np
import matplotlib.pyplot as plot
from scipy.optimize import minimize


def main():
    file_name = 'ex2data2.txt'
    X, y, m, n = read_file(file_name, ',')
    plot_points(X, y)
    X, n = map_features(X)


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
        for i in range(0, deg):
            for j in range(0, i+1):
                cur.append(np.power(X[k][0], i-j) * np.power(X[k][1], j))
        newX.append(cur)
        cur = []
    X = np.array(newX)
    return X, X.shape[1]


if __name__ == '__main__':
    main()