import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix

np.random.seed(2)

means = [[2, 2], [4,2]]
cov = [[.3, .2], [.2, .3]]

N = 30
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis = 1)
y = np.concatenate((np.ones((1,N)), -1*np.ones((1,N))), axis = 1)

X = np.concatenate((np.ones((1,2*N)), X), axis = 0)

print(y.shape)


plt.figure(figsize =(10,6))
plt.scatter(X0[0], X0[1])
plt.scatter(X1[0], X1[1], color = 'r')
plt.xlabel('X1')
plt.ylabel('X2')


def h(w, x):
    return np.sign(np.dot(w.T, x))

def has_converged(X, y, w):
    return np.array_equal(h(w,X), y)

def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    mis_points = []
    while True:
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(d,1)
            yi = y[0, mix_id[i]]
            if h(w[-1], xi) != yi:
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi*xi
                w.append(w_new)
        
        if has_converged(X, y, w[-1]):
            break
    return (w, mis_points)

d = X.shape[0]
w_init = np.random.randn(d,1)
(w, m) = perceptron(X, y, w_init)

print(w[-1])

x_vals = np.linspace(0, 6, 100)
y_vals = -(w[-1][1]/w[-1][2])*x_vals - (w[-1][0]/w[-1][2])
plt.plot(x_vals, y_vals, 'k--')
plt.show()

N_test = 5

X0_test = np.random.multivariate_normal(means[0], cov, N_test).T
X1_test = np.random.multivariate_normal(means[1], cov, N_test).T

X_test = np.concatenate((X0_test, X1_test), axis = 1)
X_test = np.concatenate((np.ones((1,2*N_test)), X_test), axis = 0)
y_test = np.concatenate((np.ones((1,N_test)), -1*np.ones((1,N_test))), axis = 1)
y_test = y_test.ravel()

y_pred = np.array([int(h(w[-1], X_test[:,x])) for x in range(X_test.shape[1])])

cnf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion matrix:')
print(cnf_matrix)

# normalized_confusion_matrix
normalized_confusion_matrix = cnf_matrix/cnf_matrix.sum(axis = 1, keepdims = True)
print('\nConfusion matrix (with normalizatrion:)')
print(normalized_confusion_matrix)
