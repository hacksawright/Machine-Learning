import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ex1data2.txt', sep=',', header=None)
df.columns = ['house_size', 'bedrooms', 'house_price']

X = np.hstack((np.ones(shape=(df.shape[0],1)), df[['house_size', 'bedrooms']]))
y = np.array(df[['house_price']])

def normal_equation (X, y):
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta
theta = normal_equation(X, y)
print(theta)

def predict(theta, data):
    data = np.hstack((np.array([1]), data))
    res = data.dot(theta)
    return res
res = predict(theta, np.array([2140,3]))
print(res)