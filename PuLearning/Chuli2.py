import pandas as pd
from sklearn.model_selection import train_test_split
import time

data = pd.read_excel('data.xlsx')

X = data.comment
y = data.sentiment
randomState = time.localtime(time.time()).tm_sec
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomState)

dt1 = pd.read_excel('data_train.xlsx')
dt2 = pd.read_excel('data_test.xlsx')
dt1['comment'] = X_train
dt1['sentiment'] = y_train
dt2['comment'] = X_test
dt2['sentiment'] = y_test
dt1.to_excel("data_train.xlsx", index=False)
dt2.to_excel("data_test.xlsx", index=False)
