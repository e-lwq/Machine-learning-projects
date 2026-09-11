from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

X,y = load_digits(return_X_y = True)



x_train, x_test, y_train, y_test = train_test_split(X,y)
mlp = MLPClassifier()
mlp = MLPClassifier(
    hidden_layer_sizes=(6,),
    max_iter=200, alpha=1e-4,
    solver='sgd',random_state=2
    )
mlp.fit(x_train, y_train)
'''
test = X[13]
plt.matshow(test.reshape(8,8), cmap=plt.cm.gray)
#plt.xticks(())
#plt.yticks(())
plt.show()

print(mlp.predict([test]))

print(mlp.score(x_test,y_test))
'''

y_pred = mlp.predict(x_test)
mask = y_pred != y_test
incorrect_x = x_test[mask]
incorrect_ypred = y_pred[mask]
incorrect_y = y_test[mask]

j=2
s = incorrect_x[j]
plt.matshow(s.reshape(8,8), cmap = plt.cm.gray)
plt.show()
print('predicted:',incorrect_ypred[j])
print('real answ:',incorrect_y[j])
