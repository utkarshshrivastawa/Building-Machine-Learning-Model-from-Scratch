import numpy as np
# code for Logistic Regression 
class Logistic_Regression():
    # declearing learning rate and no. of iteration,(Hyperparmeters) 
    def __init__(self,learning_rate , no_of_iteration):
        self.learning_rate = learning_rate
        self.no_of_iteration = no_of_iteration
    # fit function to train the model with dataset
    def fit(self,x,y):
        # number of data points in the dataset (number of rows) = m & (number of columns) = n
        self.m ,self.n = x.shape
        
    
        # initiating the wait and bias values 
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y
        #implementing Gradient Descent for Optimizaton
        for i in range(no_of_iteration):
            self.update_weights()
    
    def update_weights(self):
        # y_cap formula (sigmoid Function)
        y_cap = 1/(1+ np.exp(- (self.x.dot(self.w)+self.b)))
        # find the derivative 
        dw = (1/self.m) * np.dot(self.x.T, (y_cap - self.y))
        db = (1/self.m)*np.sum(y_cap - self.y)

        # updating the weight and bias
        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db

        # Sigmoid euation & Decision boundary 

    def predict(self,x):
        y_pred = 1/(1+ np.exp(- (self.x.dot(self.w)+self.b)))
        y_pred = np.where(y_pred>0.5 , 1 , 0)  # np.where is used to give the prediction value
        return y_pred