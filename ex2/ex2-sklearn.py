import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plot


def main():
    # polynomial features
    P = False
    file_name = 'ex2data2.txt'
    if file_name == 'ex2data2.txt':
        P = True
    X, y, m, n = read_file(file_name, ',')
    originalX = X
    # sklearn standard for y:
    ny = y.reshape((m, ))
    if P:
        poly = PolynomialFeatures(degree=6)
        X = poly.fit_transform(X)
        n = X.shape[1]
    regressor = LogisticRegression()
    regressor.fit(X, ny)
    plot_data(y, regressor, originalX, P)
    plot.show()
    while True:
        inp = input('Enter x to predict value, split by comma:\n')
        inp = inp.replace(' ', '')
        inp = inp.split(',')
        inp = list(map(float, inp))
        x = np.array([inp])
        if P:
            x = poly.fit_transform(x)
        print('result: ', regressor.predict(x))
        print('----------------------')


def read_file(file_name, delimiter):
    points = np.loadtxt(file_name, delimiter=delimiter)
    # x0s = np.ones((points.shape[0], 1))
    # points = np.append(x0s, points, axis=1)
    m = points.shape[0]
    n = points.shape[1] - 1
    return points[..., 0:n], points[..., n:], m, n


def plot_data(inpy, regressor, originalX, P):
    # Build the grid to plot:
    x_min, x_max = originalX[..., 0].min() - 0.25, originalX[..., 0].max() + 0.25
    y_min, y_max = originalX[..., 1].min() - 0.25, originalX[..., 1].max() + 0.25
    x = np.arange(x_min, x_max, 0.2)
    y = np.arange(y_min, y_max, 0.2)
    xx, yy = np.meshgrid(x, y)

    # Build the data:
    zz = np.zeros(xx.shape)
    for i in range(0, xx.shape[0]):
        for j in range(0, xx.shape[1]):
            my_x = np.array([[xx[i][j], yy[i][j]]])
            if P:
                my_x = PolynomialFeatures(degree=6).fit_transform(my_x)
            zz[i][j] = regressor.predict(my_x)

    plot.pcolormesh(xx, yy, zz)
    # plot points:
    z_x = []
    z_y = []
    o_x = []
    o_y = []
    for i in range(0, originalX.shape[0]):
        if inpy[i][0] == 1:
            o_x.append(originalX[i][0])
            o_y.append(originalX[i][1])
        else:
            z_x.append(originalX[i][0])
            z_y.append(originalX[i][1])
    plot.scatter(o_x, o_y, marker='+', c='g', label='y = 1')
    plot.scatter(z_x, z_y, marker='.', c='r', label='y = 0')




if __name__ == '__main__':
    main()