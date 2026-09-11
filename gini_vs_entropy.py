from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/eeeeeee/Desktop/spyder_folder/titanic.csv')
df['male'] = df['Sex']=='male'

X = df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
Y = df[['Survived']].values

kf = KFold(n_splits=5,shuffle=True)

for c in ['gini','entropy']:
    a=[]
    p=[]
    r=[]
    
    for train_ind, test_ind in kf.split(X):
        x_train, y_train = X[train_ind], Y[train_ind]
        x_test, y_test = X[test_ind], Y[test_ind]
        
        dt = DecisionTreeClassifier(criterion=c)
        dt.fit(x_train,y_train)
        
        y_pred = dt.predict(x_test)
        
        a.append(accuracy_score(y_test,y_pred))
        p.append(precision_score(y_test,y_pred))
        r.append(recall_score(y_test,y_pred))
        
    print(c)
    print(np.mean(a))
    print(np.mean(p))
    print(np.mean(r))
        
    print()