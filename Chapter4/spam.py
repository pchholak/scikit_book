import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

# df = pd.read_csv('data/SMSSpamCollection', delimiter='\t',
# header=None)
df = pd.read_csv('data/sms.csv')
# X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1], df[0])
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['message'],
df['label'])
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression(solver='lbfgs')
classifier.fit(X_train, y_train)
# predictions = classifier.predict(X_test)
# for i, prediction in enumerate(predictions[:5]):
#     print('Prediction: %s. Message: %s' % (prediction, X_test_raw.iat[i]))
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print('Accuracy', np.mean(scores), scores)
precisions = cross_val_score(classifier, X_train, y_train, cv=5,
scoring='precision')
print('Precision', np.mean(precisions), precisions)
recalls = cross_val_score(classifier, X_train, y_train, cv=5,
scoring='recall')
print('Recall', np.mean(recalls), recalls)
f1s = cross_val_score(classifier, X_train, y_train, cv=5,
scoring='f1')
print('F1', np.mean(f1s), f1s)
