import numpy as np
class Perceptron(object):
    """Perceptron classifier.
    Parameters
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.
    Attributes
    -----------
    w_ : 1d-array
    Weights after fitting.
    errors_ : list
    Number of misclassifications in every epoch.
    """

    def __init__(self,eta=0.01,n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def net_input(self, X):
        """Calculate the input"""
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return clss label after unit step"""
        return np.where(self.net_input(X) >= 0.0 ,1 , -1)

    def target_value(self,y):
        """Return an array containing the targets value"""
        targets = [y[0]]
        find = False
        adding_value = y[0]
        for name_set in y:
            for target in targets:
                if name_set == target:
                    find = True
            if not find:
                targets.append(name_set)
        return targets

    def convert_target(self,target,y):
        """Retrun integer that correspond at his class"""
        distinct_target_value = self.target_value(y)
        if target == distinct_target_value[0]:
            #it could be wrong the 1 it could be -1 and vice versa for
            #the else statement
            return 1
        else:
            return -1

    def fit(self, X,y):
        """Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples
        is the number of samples and
        n_features is the number of features.
        y : array-like, shape = [n_samples]
        Target values.
        Returns
        -------
        self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        """As a convention,we add an underscore to attributes that are
         not being created upon the initialization
         of the object but by calling the object's other
          methods—for example, self.w_.
        """
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                #print(self.convert_target(target,y), self.predict(xi))
                update = self.eta * (self.convert_target(target,y) - self.predict(xi))
                self.w_[1:] += update *xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
