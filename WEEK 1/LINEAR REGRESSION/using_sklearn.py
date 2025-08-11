from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ex1data1.txt', sep=',', header=None)
df.columns = ['population', 'profit']

model = LinearRegression()
model.fit(df[['population']], df[['profit']])  # train mô hình

print(model.intercept_, model.coef_)  # hệ số θ0, θ1

# Vẽ kết quả
plt.scatter(df['population'], df['profit'])
plt.plot(df['population'], model.predict(df[['population']]), c='r')
plt.xlabel('population')
plt.ylabel('profit')
plt.title('Scatter plot training data')
plt.show()
