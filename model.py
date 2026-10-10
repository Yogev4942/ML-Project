#The Model's Class
class model:
    def __init__(self):
        self.weight = 0
        self.bias = 0
    def MSE(self,y_true, y_pred):
        #the input is arrays, therefore we can use vectorized operations to calculate the mean squared error
        return ((y_true - y_pred) ** 2).mean()
    
    def calculateweightslope(self,feature,y_true,y_pred):
        return (2*feature * (y_pred - y_true)).mean()
    
    def calculatebiasslope(self,y_true,y_pred):
        return (2 * (y_pred - y_true)).mean()
    
    def newweight(self,y_true, y_pred, feature):
        self.weight = self.weight - (0.01 * self.calculateweightslope(feature,y_true,y_pred))
    def newbias(self,y_true, y_pred):
        self.bias = self.bias - (0.01 * self.calculatebiasslope(y_true,y_pred))

    def predict(self, feature):
        return self.weight * feature + self.bias

    def fit(self, feature, y_true, epochs=1000):
        for _ in range(epochs):
            y_pred = self.predict(feature)
            self.newweight(y_true, y_pred, feature)
            self.newbias(y_true, y_pred)