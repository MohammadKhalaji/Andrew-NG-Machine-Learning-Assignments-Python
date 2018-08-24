import numpy as np
import matplotlib.pyplot as plot
from scipy.optimize import minimize


def main():
    file_name = 'ex2data1.txt'
    X, y, m, n = read_file(file_name, ',')
    theta = np.zeros((n, 1))
    theta = minimize(lambda t: cost(X, y, t), theta, method='Nelder-Mead').x.reshape((n, 1))
    print('Training accuracy: %{0}'.format(get_accuracy(X, y, theta)))
    plot_decision(X, theta)
    plot_points(X, y)
    plot.xlabel('Exam 1 Score')
    plot.ylabel('Exam 2 Score')
    plot.title('Logistic Regression - Not Regularized\nNon-Polynomial Features  ')
    plot.legend()
    plot.savefig('ex2-NotRegularized.png', dpi=300)
    plot.show()
    while True:
        inp = '1,'
        inp += input('Enter x to predict value, split by comma:\n')
        inp = inp.replace(' ', '')
        inp = inp.split(',')
        inp = list(map(float, inp))
        x = np.array([inp]).T
        print('result: ', predict(x, theta))
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
    plot.scatter(o_x, o_y, marker='+', c='g', label='Admitted')
    plot.scatter(z_x, z_y, marker='.', c='r', label='Not Admitted')


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


def plot_decision(X, theta):
    # Build the grid to plot:
    x_min, x_max = X[..., 1].min() - 1, X[..., 1].max() + 1
    y_min, y_max = X[..., 2].min() - 1, X[..., 2].max() + 1
    x = np.arange(x_min, x_max, 0.05)
    y = np.arange(y_min, y_max, 0.05)
    xx, yy = np.meshgrid(x, y)

    # Build the data:
    zz = np.zeros(xx.shape)
    for i in range(0, xx.shape[0]):
        for j in range(0, xx.shape[1]):
            my_x = np.array([[1, xx[i][j], yy[i][j]]]).T
            zz[i][j] = predict(my_x, theta)

    plot.pcolormesh(xx, yy, zz)


if __name__ == '__main__':
    main()
