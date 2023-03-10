import numpy as np


class LinearRegression:

    def __init__(self, lr=0.001, n_iter=1000):
        # define weights and bias
        self.weights = None
        self.bias = None
        # for gradient descent you need learning rate and number of iterations to try to reach minimum
        self.lr = lr
        self.n_iter = n_iter

    # Every algorithm needs fit and predict methods, so we will be defining them in this class
    # Requires input features and the corresponding outputs to find best fit line
    # Tip: X is a matrix (n_samples, n_features) and y is a vector (n_samples, 1)
    # weights will be n_features x 1
    # even during prediction X size will be 1 x n_features, but y will be n_samples x 1
    def fit(self, X, y):
        # Step 1: Need to default weights and bias to 0
        # Since we have as many weights as number of features (y=wX+b), so weight would be a 1D array (vector)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # n_features x 1 size
        self.bias = 0  # just 0 as only 1 bias in the equation

        # Step 2: Perform gradient descent
        # loop through all iterations
        for n in range(self.n_iter):
            # 2.1 write down the equation
            # note that using a dot product of features matrix with weights vector (result will be wx for each sample)
            # the returned y_predicted will be a vector of length equal to number of samples
            y_predicted = self.predict(X)

            # 2.2 calculate error
            error = y_predicted - y

            # 2.2 calculate gradients - equation is derivative of cost function wrt weight and bias
            # note that we need to transpose X here because
            # we need multiplication of first sample of feature with first sample's error
            dw = (2 / n_samples) * np.dot(X.T, error)
            # equation is simple wrt bias and is just 2*avg sum of errors
            db = (2 / n_samples) * np.sum(error)

            # 2.3 update weights
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    # predict method to predict to test data - need features as input
    def predict(self, X):
        # note that weights is second param
        prediction = np.dot(X, self.weights) + self.bias
        return prediction
