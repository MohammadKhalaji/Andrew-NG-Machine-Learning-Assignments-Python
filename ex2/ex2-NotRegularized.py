import numpy as np
import matplotlib.pyplot as plot


def main():
    file_name = 'ex2data1.txt'
    X, y, m, n = read_file(file_name, ',')
    plot_points(X, y)


def read_file(file_name, delimiter):
    points = np.loadtxt(file_name, delimiter=delimiter)
    x0s = np.ones((points.shape[0], 1))
    points = np.append(x0s, points, axis=1)
    m = points.shape[0]
    n = points.shape[1] - 1
    return points[..., 0:n], points[..., n:], m, n


def plot_points(X, y):
    z_x = []
    z_y = []
    o_x = []
    o_y = []
    for i in range(0, X.shape[0]):
        if y[i][0] > 0.5:
            o_x.append(X[i][1])
            o_y.append(X[i][2])
        else:
            z_x.append(X[i][1])
            z_y.append(X[i][2])
    plot.scatter(o_x, o_y, marker='+', c='g')
    plot.scatter(z_x, z_y, marker='.', c='r')


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def h(X, theta):
    return sigmoid(np.matmul(X, theta))


def cost(X, y, theta):
    m = X.shape[0]
    ans = np.matmul(y.T, np.log(h(X, theta))) + np.matmul(1 - y, np.log(1 - h(X, theta)))
    return (-1 / m) * ans


def gradient(X, y, theta):
    m = X.shape[0]
    ans = h(X, theta) - y
    ans = np.matmul(X.T, ans)
    return (1 / m) * ans


if __name__ == '__main__':
    main()
