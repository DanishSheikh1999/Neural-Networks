
class Preceptron:
    def __init__(self, learning_rate=.5):
        self.learning_rate = learning_rate

    def activation(self, output):
        if output > 0:
            return 1
        else:
            return 0

    def intialize_parameters(self):
        W = [[0]*self.n_x for i in range(self.n_y)]
        b = [0]*self.n_y
        return W, b

    def layer_sizes(self, X, Y):
        n_x = len(X[0])
        n_y = len(Y[0])
        m = len(X)
        return (m, n_x, n_y)

    def linear_forward(self, index, X):
        output = self.bias[index]
        for i in range(len(self.weight[index])):
            output += self.weight[index][i] * X[i]

        return self.activation(output)

    def updateParameters(self, index, X, suppOutput, actualOutput):
        for i in range(len(self.weight[index])):
            self.weight[index][i] += round(self.learning_rate *
                                           (actualOutput - suppOutput)*X[i], 1)
        self.bias[index] += self.learning_rate*(actualOutput - suppOutput)

    def predict(self, X):
        pred = [0]*self.n_y
        for i in range(self.n_y):
            pred[i] = self.linear_forward(i, X)
        return pred

    def fit(self, X, y):
        (self.m, self.n_x, self.n_y) = self.layer_sizes(X, y)
        predicted_output = [[0]*self.n_y for i in range(self.m)]
        self.weight, self.bias = self.intialize_parameters()
        n = 0
        while predicted_output != y and n < 10000:
            n += 1
            for j in range(self.m):
                for i in range(self.n_y):
                    predicted_output[j][i] = self.linear_forward(
                        i, X[j])
                    if predicted_output[j][i] != y[j][i]:
                        self.updateParameters(
                            i, X[j], predicted_output[j][i], y[j][i])
        return {"W":self.weight,"b":self.bias}

