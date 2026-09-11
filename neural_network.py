from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

x,y = make_classification(n_samples = 1000,n_features = 2, n_redundant = 0, n_informative = 2, random_state = 3)
#print(x)
#print(y)
#print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 3)

'''plt.scatter(x[y==0][:,0], x[y==0][:,1], s=100, edgecolors = 'k')
plt.scatter(x[y==1][:,0], x[y==1][:,1], s=100, edgecolors = 'k', marker ='^')
plt.show()

plt.scatter(x[:,0], x[:,1], c = y)'''

#mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes = (100,50))
#mlp.fit(x_train,y_train)

'''param_grid = {
    'max_iter': [1000],
    'hidden_layer_sizes': [(100), (100,50), (100,50,50)],
    'solver': ['lbfgs', 'adam', 'sgd']
    }
mlp = MLPClassifier()
gs = GridSearchCV(mlp, param_grid, cv = 5)
gs.fit(x_train, y_train)
print(gs.best_params_)'''

#print(mlp.score(x_test,y_test))

mlp = MLPClassifier(max_iter = 1000, hidden_layer_sizes = 100, solver = 'sgd')
mlp.fit(x_train,y_train)
print(mlp.score(x_test,y_test))


'''
** mlp uses gradient descent to find coeff s.t. loss function is optimal
hyperparameters:
    - max_iter
    - hidden_layer_sizes
    - alpha (step size; decrease alpha --> increase max_iter)
    - solver (algorithm to find optimal soln)
        - lbfgs: for small datasets
        - sgd
        - adam (default): for large datasets
'''