import numpy as np
import matplotlib.pyplot as plot
import statistics as stats


NORMALIZE = True # Must be true for ex1data2.txt


def main():
    file_name = 'ex1data2.txt'
    # X, y : training data
    # m : no. of training data
    # n : no. of features, including x0 = 1
    X, y, m, n = read_file(file_name, delimiter=',')
    feature_stats = None
    if NORMALIZE:
        X, feature_stats = normalize(X)
    theta = np.zeros((n, 1))
    theta = gradient_descent(X, y, theta, iters=10000, alpha=0.01)
    print(theta)

    if file_name == 'ex1data1.txt':
        plot_points(X, y)
        plot_line(X, theta)
        plot.show()

    while True:
        inp = '1,'
        inp += input('Enter x to predict value, split by comma:\n')
        inp = inp.replace(' ', '')
        inp = inp.split(',')
        inp = list(map(float, inp))
        x = np.array([inp])
        predict(x.T, theta, feature_stats)


def read_file(file_name, delimiter=','):
    points = np.loadtxt(file_name, delimiter=delimiter)
    x0s = np.ones((points.shape[0], 1))
    points = np.append(x0s, points, axis=1)
    m = points.shape[0]
    n = points.shape[1] - 1
    return points[..., 0:n], points[..., n:], m, n


def cost(X, y, theta):
    # cost = (1/2m) ((Xtheta- y)T * (Xtheta - y))
    m = X.shape[0]
    ans = np.matmul(X, theta)
    ans = ans - y
    ans = np.matmul(ans.T, ans)
    return (1/(2*m)) * ans


# only works for linear regression with 2 variables ( 1 feature )
def plot_points(X, y):
    plot.scatter(X.T[1], y.T, c='b')
    # plot.show()


def plot_line(X, theta):
    ys = np.matmul(X, theta)
    ys = ys.T
    ys = [y for y in ys[0]]
    xs = X.T[1:, ...]
    xs = [x for x in xs[0]]
    plot.plot(xs, ys, 'r')
    # plot.show()


def gradient(X, y, theta):
    # gradient = (1/m) * ((XT) * (Xtheta - y))
    ans = np.matmul(X, theta) - y
    ans = np.matmul(X.T, ans)
    m = X.shape[0]
    return (1/m) * ans


def gradient_descent(X, y, theta, iters=1500, alpha=0.01):
    for i in range(iters):
        theta = theta - alpha * gradient(X, y, theta)
    return theta


def normalize(X):
    res = []
    X = X.T
    res.append(X[0])
    feature_stats = [{'mean': 1, 'stdev': 0}]
    for i in range(1, X.shape[0]):
        mean = stats.mean(X[i])
        stdev = stats.stdev(X[i])
        res.append((X[i] - mean) / stdev)
        feature_stats.append({'mean': mean, 'stdev': stdev})
    return np.array(res).T, feature_stats


def predict(x, theta, feature_stats):
    if feature_stats is not None:
        for i in range(1, x.shape[0]):
            x[i][0] = (x[i][0] - feature_stats[i]['mean']) / feature_stats[i]['stdev']
    print('result: ', np.matmul(theta.T, x)[0][0])
    print()



if __name__ == '__main__':
    main()