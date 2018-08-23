import numpy as np
import matplotlib.pyplot as plot
from scipy.optimize import minimize


def main():
    file_name = 'ex2data1.txt'
    X, y, m, n = read_file(file_name, ',')
    plot_points(X, y)
    theta = np.zeros((n, 1))
    theta = minimize(lambda t: cost(X, y, t), theta, method='Nelder-Mead').x.reshape((n, 1))
    print('Training accuracy: %{0}'.format(get_accuracy(X, y, theta)))
    plot_decision_boundary(X, theta)
    plot.show()
    while True:
        inp = '1,'
        inp += input('Enter x to predict value, split by comma:\n')
        inp = inp.replace(' ', '')
        inp = inp.split(',')
        inp = list(map(float, inp))
        x = np.array([inp]).T
        print('resutl: ', predict(x, theta))
        print('----------------------')



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
        if y[i][0] == 1:
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
    ans = np.matmul(y.T, np.log(h(X, theta))) + np.matmul((1 - y).T, np.log(1 - h(X, theta)))
    return (-1 / m) * ans


def gradient(X, y, theta):
    m = X.shape[0]
    ans = h(X, theta) - y
    ans = np.matmul(X.T, ans)

    return (1 / m) * ans


def predict(x, theta):
    ans = np.matmul(theta.T, x)[0][0]
    if ans >= 0:
        return 1
    return 0


def get_accuracy(X, y, theta):
    m = X.shape[0]
    corrects = 0
    for i in range(0, m):
        x = np.array([X[i]]).T
        if predict(x, theta) == y[i]:
            corrects += 1
    return 100 * corrects / m


def plot_decision_boundary(X, theta):
    xs = X.T[1]
    t0 = theta[0][0]
    t1 = theta[1][0]
    t2 = theta[2][0]
    ys = [(-t0 - t1*x)/t2 for x in xs]
    plot.plot(xs, ys, c='b')


if __name__ == '__main__':
    main()
