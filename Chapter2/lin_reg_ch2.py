# from sklearn.linear_model import LinearRegression
# # Training data
# X = [[6], [8], [10], [14], [18]]
# y = [[7], [9], [13], [17.5], [18]]
# # Create and fit the model
# model = LinearRegression()
# model.fit(X, y)
# print('A 12" pizza should cost: $%.2f' % model.predict([[12]])[0])
#
# import numpy as np
# print('Residual sum of squares: %.2f' % np.mean((model.predict(X) - y) ** 2))
#
# print(np.var(X, ddof=1))
# print(np.cov([6, 8, 10, 14, 18], [7, 9, 13, 17.5, 18])[0][1])

# from sklearn.linear_model import LinearRegression
# X = [[6], [8], [10], [14], [18]]
# y = [[7], [9], [13], [17.5], [18]]
# X_test = [[8], [9], [11], [16], [12]]
# y_test = [[11], [8.5], [15], [18], [11]]
# model = LinearRegression()
# model.fit(X, y)
# print('R-squared: %.4f' % model.score(X_test, y_test))

# from numpy.linalg import inv
# from numpy import dot, transpose
# X = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
# y = [[7], [9], [13], [17.5], [18]]
# print(dot(inv(dot(transpose(X), X)), dot(transpose(X), y)))
#
# from numpy.linalg import lstsq
# print(lstsq(X, y)[0])

from sklearn.linear_model import LinearRegression
X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7], [9], [13], [17.5], [18]]
model = LinearRegression()
model.fit(X, y)
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11], [8.5], [15], [18], [11]]
predictions = model.predict(X_test)

for i, prediction in enumerate(predictions):
    print('Predicted: %s, Target: %s' % (prediction, y_test[i]))
print('R-squared: %.2f' % model.score(X_test, y_test))
