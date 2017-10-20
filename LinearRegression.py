import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 10)
#inline method y = x
f = lambda x: x
#Turn f into numpy thing
F = np.vectorize(f)
Y = F(X)

#init dataset
num = 100
random_sign = np.vectorize(lambda x: x if np.random.sample() > 0.5 else -x)
data_X = np.linspace(1, 9, num)
data_Y = random_sign(np.random.sample(num) * 2) + F(data_X)

#Linear regression
from sympy import *
def linear_regression(X, Y):
    a, b = symbols('a b')
    residual = 0 #殘值
    for i in range(num):
        residual += (Y[i] - (a * X[i] + b)) ** 2
    #diff a and b respectively
    f1 = diff(residual, a)
    f2 = diff(residual, b)
    res = solve([f1, f2], [a, b])
    return res[a], res[b]
a, b = linear_regression(data_X, data_Y)

LR_X = X
h = lambda x: a * x + b
H = np.vectorize(h)
LR_Y = H(LR_X)
#plt.plot(X, Y, 'b')
plt.plot(LR_X, LR_Y, 'g')
plt.plot(data_X, data_Y, 'ro')
#Data test
#Define 4 extraordinary data points and 3 normal data points
DataSet = [[1.3, 10], [2.6, 8], [3.8, 7], [4.5, 9], [3.2, 3.0], [2.1, 1.9], [1.4, 1.3]]
#DataSet = [[1.3, 2.6, 3.8, 4.5, 3.2, 2.1, 1.4], [10, 8, 7, 9, 3.0, 1.9, 1.3]]
i = 0
while i < len(DataSet):
    plt.plot(DataSet[i][0], DataSet[i][1], 'bo')
    i += 1
plt.show()

count = 0
for i in range(len(data_X)):
    if(data_X[i] < 5):
        count += 1
ordinary = (count + 3) / (float)(count + len(DataSet))
exception = (count + 4) / (float)(count + len(DataSet))
print (format(ordinary, '.2f'))
print (format(exception, '.2f'))