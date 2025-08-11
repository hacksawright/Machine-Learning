import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ex1data1.txt', sep=',', header=None)
df.columns = ['population', 'profit']

X = np.hstack((np.ones(shape=(df.shape[0],1)), df[['population']]))
y = np.array(df[['profit']])

def normal_equation(X, y):
    theta = np.linalg.inv((X.T.dot(X))).dot(X.T).dot(y)
    return theta
theta = normal_equation(X, y)
print(theta)

plt.scatter(df[['population']], df[['profit']])
plt.plot(X[:, 1], X.dot(theta), c ='r')
plt.xlabel('population')
plt.ylabel('profit')
plt.title('Scatter plot training data')
plt.show()