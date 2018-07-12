# Try classifying classes 1 and 2 from the iris dataset with SVMs,
# with the 2 first features. Leave out 10% of each class and test
# prediction performance on these observations.

from sklearn import datasets, svm
import numpy as np
def exctact_first_2_feature(X):
    result = []
    for i in range(len(X)):
        result.append(X[i][:2])
    return result

def take_2_class(X,y):
    ts = []
    target = []
    for i in range(len(X)):
        if(y[i]!=0):
            ts.append(X[i])
            target.append(y[i])
    return ts,target

def shuffle(X,y):
    X_copy = X
    y_copy = y
    np.random.seed(0)
    indices = np.random.permutation(len(X))
    for i in range(len(X)):
        X[i] = X_copy[indices[i]]
        y[i] = y_copy[indices[i]]

def rate_correct(X,Y):
    total_correct = 0
    for i in range(len(X)):
        if(int(X[i])==int(Y[i])):
            total_correct += 1
    return total_correct/len(X)

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
fst_feature = exctact_first_2_feature(iris_X)
iris = take_2_class(fst_feature,iris_y)
iris_X = iris[0]
iris_y = iris[1]
shuffle(iris_X,iris_y)
svc = svm.SVC(kernel='linear')
# linear,poly , rbf type of kernel
n_samples = len(iris_X)

X_train = iris_X[:int(.9 * n_samples)]
y_train = iris_y[:int(.9 * n_samples)]
X_test = iris_X[int(.9 * n_samples):]
y_test = iris_y[int(.9 * n_samples):]
svc.fit(X_train,y_train)
print("Rate: ",rate_correct(svc.predict(X_test),y_test)*100,"%" )
