import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures


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


if __name__ == '__main__':
    main()