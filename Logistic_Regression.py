class regression_algo:
    
    #Initilisation
    def __init__(self, lr=0.01, iter=100000, fit_intercept=True):
        self.lr = lr
        self.iter = iter
        self.fit_intercept = fit_intercept
        
    #Intercept
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    #Sigmoid function
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    #Loss Function
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    #Fitting data into the model
    def fit_data(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)        
        # weights initialization
        self.theta = np.zeros(X.shape[1])        
        for i in range(self.iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient            
            if(i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')
    
    #Predict Probability
    def predict_probability(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X) 
        return self.__sigmoid(np.dot(X, self.theta))
    
    #Predict function to test the model on test data
    def predict(self, X, threshold=0.5):
        return self.predict_probability(X) >= threshold


    #Apply model on training data with 0.1 as learning rate. 
	model = regression_algo(lr=0.1, iter=300000)
	model.fit_data(X_train, Y_train)