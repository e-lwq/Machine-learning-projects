import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/eeeeeee/Desktop/spyder_folder/csv_of_1000img.csv')

feature_names = []
for i in range(2500):
    feature_names.append(str(i))
X = df[feature_names].values
y = df['target'].values


X_train, X_test, y_train, y_test = train_test_split(X,y)

mlp = MLPClassifier(max_iter = 5000)
mlp.fit(X_train,y_train)

print(mlp.score(X_test,y_test))