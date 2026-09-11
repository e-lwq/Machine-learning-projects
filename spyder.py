import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.metrics import classification_report as cr
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, roc_auc_score

df = pd.read_csv('/Users/eeeeeee/Desktop/spyder_folder/titanic.csv')
df['male'] = df['Sex'] == 'male'

model = LogisticRegression()

x = df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
y = df['Survived'].values

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=10)

#print(x.shape, y.shape)
#print(x_train.shape, y_train.shape)
#print(x_test.shape, y_test.shape)

model.fit(x_train,y_train)
#print(model.score(x_test,y_test))

#y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test,y_pred_proba)

plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],linestyle = '-.')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('1-specificity')
plt.ylabel('sensitivity')
plt.show()

print(roc_auc_score(y_test,y_pred_proba))


model2 = LogisticRegression(solver='liblinear')
x2 = df[['Pclass','Age']].values
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y, random_state=10)
model2.fit(x_train, y_train)
y_pred_proba2 = model2.predict_proba(x_test)[:,1]
#fpr2, tpr2, thresholds2 = roc_curve(y_test2,y_pred_proba2)
print(roc_auc_score(y_test,y_pred_proba2))

'''
print("accuracy:",accuracy_score(y_test,y_pred))
print("precision:",precision_score(y_test,y_pred))
print("recall:",recall_score(y_test,y_pred))
print("f1:",f1_score(y_test,y_pred))

sensitivity = recall_score
print(sensitivity(y_test,y_pred))

print(precision_recall_fscore_support(y_test,y_pred))
'''


'''
print(cr(y,y_pred))

print(confusion_matrix(y,y_pred))

#print((y==y_pred).sum()/y.shape[0])
#print(sklearn.metrics.accuracy_score(y,y_pred))
'''
'''
a = model.coef_[0][0]
b = model.coef_[0][1]
c = model.intercept_[0]

print(a,b,c)
plt.scatter(df['Fare'],df['Age'],c=df['Survived'])
plt.plot([0,-c/b],[-c/a,0])
plt.xlabel('Fare')
plt.ylabel('Age')'''
