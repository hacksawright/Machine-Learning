import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(2)

means = [[2, 2], [4,2]]
cov = [[.3, .2], [.2, .3]]

N = 20
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis = 1)
y = np.concatenate((np.ones((1,N)), -1*np.ones((1,N))), axis = 1)

X = np.concatenate((np.ones((1,2*N)), X), axis = 0)


def h(w, x):
    return np.sign(np.dot(w.T, x))

def has_converged(X, y, w):
    return np.array_equal(h(w,X), y)

def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    mis_points = []

    plt.figure(figsize=(10, 6))
    plt.scatter(X0[0], X0[1], color='b', label='Class +1')
    plt.scatter(X1[0], X1[1], color='r', marker='^', label='Class -1')
    plt.xlabel('X1')
    plt.ylabel('X2')

    boundary_line = None  # để lưu đường hiện tại

    while True:
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(d, 1)
            yi = y[0, mix_id[i]]
            if h(w[-1], xi) != yi:
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi * xi
                w.append(w_new)

                # # Xoá đường cũ nếu tồn tại
                if boundary_line is not None:
                    boundary_line.remove()

                # # Vẽ đường mới
                x_vals = np.linspace(min(X[1,:])-1, max(X[1,:])+1, 100)
                y_vals = -(w[-1][1] / w[-1][2]) * x_vals - (w[-1][0] / w[-1][2])
                boundary_line, = plt.plot(x_vals, y_vals, 'k--')

                all_y = np.concatenate([X[2,:], y_vals])
                plt.xlim(min(X[1,:])-1, max(X[1,:])+1)
                plt.ylim(min(all_y)-1, max(all_y)+1)

                plt.pause(0.7)
                # print(w_new, "hi")

        if has_converged(X, y, w[-1]):
            break

    plt.legend()
    plt.show()
    return (w, mis_points)


d = X.shape[0]
w_init = np.random.randn(d,1)
(w, m) = perceptron(X, y, w_init)

print(w[-1])
