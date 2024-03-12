import numpy as np
class Linear_Regression():
    # initiating the parameters {learing_rat & no_of_iterations}
    def __init__(self,learning_rate , no_of_iterations):
        
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        
    def  fit(self,x,y):
        # no of training examples = m & no. of features = n
        self.m ,self.n = x.shape # it is no. of rows and columns 

        # initiating the weight and bias 
        self.w= np.zeros(self.n)
        self.b = 0

        
        self.x = x
        self.y = y
        # implementing Gradient Descent
        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self,):
        Y_prediction = self.predict(self.x)
        
        # calculate the gradient
        
        dw = -(2*(self.x.T).dot(self.y-Y_prediction))/self.m
        db = - 2*np.sum(self.y-Y_prediction)/self.m
        
        # updating the weights
        
        self.w = self.w-self.learning_rate*dw
        self.b = self.b-self.learning_rate*db
        
    def predict(self,x):
        return x.dot(self.w)+ self.b
        