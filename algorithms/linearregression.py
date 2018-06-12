from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_train = [1, 3, 5, 1, 8, 1, 1, 16, 1, 20]

x_train = np.reshape(np.array(x_train), (-1, 1))
y_train = np.reshape(np.array(y_train), (-1, 1))

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
linear.score(x_train, y_train)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

plt.plot(x_train, linear.predict(x_train), color = '#000000')
plt.scatter(x_train, y_train)
plt.show()
