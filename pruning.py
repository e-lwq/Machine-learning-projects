import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('/Users/eeeeeee/Desktop/spyder_folder/titanic.csv')
df['male'] = df['Sex']=='male'
x = df[['Pclass','male','Age','Fare','Siblings/Spouses','Parents/Children']].values
y=df[['Survived']].values

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=10)

param_grid = {
    'max_depth':[5,15,25],
    'min_samples_leaf':[1,3],
    'max_leaf_nodes':[10,20,35,50]}

dt = DecisionTreeClassifier()
gs = GridSearchCV(dt, param_grid, scoring='f1', cv=5)
gs.fit(x,y)
print(gs.best_params_)
print(gs.best_score_)
