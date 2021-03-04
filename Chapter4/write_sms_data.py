import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv('data/SMSSpamCollection', delimiter='\t',
header=None)

labs = np.zeros((len(df[0]),))
for i, lab in enumerate(df[0]):
    if df[0][i] == 'spam':
        labs[i] = 1

df1 = pd.DataFrame({'message': df[1],
                    'label': labs})
# print(df1)
df1.to_csv('data/sms.csv', sep=',')
