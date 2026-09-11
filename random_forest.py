import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data['data'], columns = cancer_data['feature_names'])
df['target'] = cancer_data['target']

x=df[cancer_data.feature_names].values
y=df['target'].values

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=10)

rf = RandomForestClassifier(random_state = 10)

n_estimators = list(range(1,21))
grid_param = {
    'n_estimators':n_estimators}

gs = GridSearchCV(rf,grid_param,cv=5)
gs.fit(x,y)

print(gs.best_params_)
print(gs.best_score_)

scores = gs.cv_results_['mean_test_score']

plt.plot(n_estimators, scores)
plt.xlim(0,20)
plt.ylim(0.9,1)
plt.show()

rf2 = RandomForestClassifier(n_estimators = 10)
rf2.fit(x_train,y_train)
print(rf2.score(x_test,y_test))