import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("ex2data1.txt", header=None)
X = df.iloc[:, :-1].values              #storing values (first 2)
y = df.iloc[:, -1].values               #storing values (last- binary wali)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costFunction(theta, X, y):
    m = len(y)
    predictions = sigmoid(np.dot(X, theta))
    print(predictions)
    error = (-y * np.log(predictions)) - ((1 - y) * np.log(1 - predictions))
    cost = 1 / m * sum(error)
    grad = 1 / m * np.dot(X.transpose(), (predictions - y))

    return cost[0], grad


def featureNormalization(X):
    mean = np.mean(X, axis=0)       #mean calculated along the column
    std = np.std(X, axis=0)         #std calculated along the column
    X_norm = (X - mean) / std

    return X_norm, mean, std


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        cost, grad = costFunction(theta, X, y)
        theta = theta - (alpha * grad)
        J_history.append(cost)

    return theta, J_history













student_in, student_out = (y == 1).reshape(100, 1), (y == 0).reshape(100, 1)        #array of size 100 reshaped 100rows, 1 column, y == 1 prints seedhi
                                                                                    #aids us in plotting the two separate classes
m, n = X.shape[0], X.shape[1]
                                                                                    #m has number of rows, n has number of columns
                                                                                    #X has complete data except 0 and 1
#X, X_mean, X_std = featureNormalization(X)                                          #returning Normalized X, mean and std
X = np.append(np.ones((m, 1)), X, axis=1)
y = y.reshape(m, 1)                                                                 #reshaped into a single column

initial_theta = np.zeros((n + 1, 1))
cost, grad = costFunction(initial_theta, X, y)
print("Initial theta's cost is = ", cost)
print("Gradient at initial theta (zeros):", grad)
#
theta, J_history = gradientDescent(X, y, initial_theta, 1, 400)
#
print("Theta optimized by gradient descent:", theta)
print("The cost of the optimized theta:", J_history[-1])
#
# plt.scatter(X[student_in[:, 0], 1], X[student_in[:, 0], 2], c="r", marker="+", label="Admitted")
# plt.scatter(X[student_out [:, 0], 1], X[student_out[:, 0], 2], c="b", marker="x", label="Not admitted")
# x_value = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
# y_value = -(theta[0] + theta[1] * x_value) / theta[2]
# plt.plot(x_value, y_value, "g")
# plt.xlabel("Exam 1 score")
# plt.ylabel("Exam 2 score")
# plt.legend(loc=0)
#
#
# # x_test = np.array([45, 85])
# # x_test = (x_test - X_mean) / X_std
# # x_test = np.append(np.ones(1), x_test)
# # prob = sigmoid(x_test.dot(theta))
# #print("For a student with scores 45 and 85, we predict an admission probability of", prob[0])
# #plt.show()