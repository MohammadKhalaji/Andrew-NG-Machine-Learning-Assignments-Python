import numpy as np
import sklearn.preprocessing
import sklearn.linear_model

NORMALIZE = False


def main():
    file_name = 'ex1data1.txt'
    X, y, m, n = read_file(file_name, delimiter=',')
    if NORMALIZE:
        X = sklearn.preprocessing.scale(X)

    regressor = sklearn.linear_model.LinearRegression()
    regressor.fit(X, y)


    print('theta0: ')
    print(regressor.intercept_)
    print('coefficients: ')
    print(regressor.coef_)
    print()

    while True:
        inp = input('Enter x to predict value, split by comma:\n')
        inp = inp.replace(' ', '')
        inp = inp.split(',')
        inp = list(map(float, inp))
        x = np.array([inp])
        if NORMALIZE:
            x = sklearn.preprocessing.scale(x)
        print(regressor.predict(x)[0][0])
        print()


def read_file(file_name, delimiter):
    points = np.loadtxt(file_name, delimiter=delimiter)
    # x0s = np.ones((points.shape[0], 1))
    # points = np.append(x0s, points, axis=1)
    m = points.shape[0]
    n = points.shape[1] - 1
    return points[..., 0:n], points[..., n:], m, n



if __name__ == '__main__':
    main()
