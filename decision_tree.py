from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/eeeeeee/Desktop/spyder_folder/titanic.csv')
df['male'] = df['Sex']=='male'

x = df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
y = df[['Survived']].values

kf = KFold(n_splits=5,shuffle=True,random_state=10)
splits = kf.split(x)

a=[]
p=[]
r=[]
f=[]

a2=[]
p2=[]
r2=[]
f2=[]

for train_ind, test_ind in splits:
    x_train, y_train = x[train_ind], y[train_ind]
    x_test, y_test = x[test_ind], y[test_ind]
    
    dtmodel = DecisionTreeClassifier(criterion = 'entropy')
    dtmodel.fit(x_train,y_train)
    
    y_pred = dtmodel.predict(x_test)
    
    a.append(accuracy_score(y_test,y_pred))
    p.append(precision_score(y_test,y_pred))
    r.append(recall_score(y_test,y_pred))
    f.append(f1_score(y_test,y_pred))
    
    
    lrmodel = LogisticRegression(solver='liblinear')
    lrmodel.fit(x_train,y_train)
    
    y_pred2 = lrmodel.predict(x_test)
    
    a2.append(accuracy_score(y_test,y_pred2))
    p2.append(precision_score(y_test,y_pred2))
    r2.append(recall_score(y_test,y_pred2))
    f2.append(f1_score(y_test,y_pred2))
    
print('Decision Tree:')
print(np.mean(a))
print(np.mean(p))
print(np.mean(r))
#print(np.mean(f))

print('Logistic Regression:')
print(np.mean(a2))
print(np.mean(p2))
print(np.mean(r2))
#print(np.mean(f2))
    