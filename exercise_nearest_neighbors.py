#Try classifying the digits dataset with nearest neighbors and
# a linear model. Leave out the last 10% and test prediction performance
# on these observations.

from sklearn import datasets, neighbors, linear_model
import numpy as np

def rate_correct(X,Y):
    total_correct = 0
    for i in range(len(X)):
        if(int(X[i])==int(Y[i])):
            total_correct += 1
    return total_correct/len(X)

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
percent_10_tot = int(len(X_digits)*10/100)

np.random.seed(0)
indices = np.random.permutation(len(X_digits))
# indices is an array containing the permutation of the number from 1 to
# legnth of X_digits

digits_train_X = X_digits[indices[:-percent_10_tot]]
digits_train_y = y_digits[indices[:-percent_10_tot]]
digits_test_X = X_digits[indices[len(X_digits)-percent_10_tot:]]
digits_test_y = y_digits[indices[len(X_digits)-percent_10_tot:]]

k_n_n = neighbors.KNeighborsClassifier()
k_n_n.fit(digits_train_X,digits_train_y)

regr = linear_model.LogisticRegression()
regr.fit(digits_train_X,digits_train_y)
print("rate correcteness")
print("KNN: ", int(rate_correct(k_n_n.predict(digits_test_X),digits_test_y)*100),"%")
print("Regressor: ", int(rate_correct(regr.predict(digits_test_X),digits_test_y)*100),"%")
