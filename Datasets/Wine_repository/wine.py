# import pandas as pd
# df = pd.read_csv('winequality-red.csv', sep=';')
# print(df.corr())

# import matplotlib.pylab as plt
# plt.scatter(df['alcohol'], df['quality'])
# plt.xlabel('Alcohol')
# plt.ylabel('Quality')
# plt.title('Alcohol against Quality')
# plt.show()

# from sklearn.linear_model import LinearRegression
# import pandas as pd
# import matplotlib.pylab as plt
# from sklearn.model_selection import train_test_split
# # from sklearn.cross_validation import train_test_split
#
# df = pd.read_csv('winequality-red.csv', sep=';')
# X = df[list(df.columns)[:-1]]
# y = df['quality']
# X_train, X_test, y_train, y_test = train_test_split(X, y)
#
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
# y_predictions = regressor.predict(X_test)
# print('R-squared:', regressor.score(X_test, y_test))

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
df = pd.read_csv('winequality-red.csv', sep=';')
X = df[list(df.columns)[:-1]]
y = df['quality']
regressor = LinearRegression()
scores = cross_val_score(regressor, X, y, cv=5)
print(scores.mean(), scores)
