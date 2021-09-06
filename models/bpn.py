import numpy as np


class BPN:

    def __init__(self, hidden_neurons, learning_rate=0.1, num_epochs=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_epochs
        self.n_h = hidden_neurons

    def intialize_weights(self, nx, nh, ny):
        np.random.seed(1)
        W1 = np.random.randn(nh, nx)*.01
        b1 = np.random.randn(nh, 1)*.01
        W2 = np.random.randn(ny, nh)*.01
        b2 = np.random.randn(ny, 1)*.01

        assert (W1.shape == (nh, nx))
        assert (b1.shape == (nh, 1))
        assert(W2.shape == (ny, nh))
        assert (b2.shape == (ny, 1))

        return {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2
        }

    def sigmoid(self, Z):
        A = 1/(1+np.exp(-1*Z))
        return A

    def forward_propagation(self, X):

        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]
        Z1 = np.dot(W1, X)+b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1)+b2
        A2 = self.sigmoid(Z2)
        assert(A2.shape == (4, X.shape[1]))

        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}

        return A2, cache

    def compute_cost(self, A2, Y):

        m = Y.shape[1]
        logprobs = np.multiply(np.log(A2), Y)+np.multiply(np.log(1-A2), 1-Y)
        cost = -np.sum(logprobs)/m
        cost = float(np.squeeze(cost))
        assert(isinstance(cost, float))

        return cost

    def layer_sizes(self, X, Y):
        n_x = X.shape[0]
        n_y = Y.shape[0]
        return (n_x, n_y)

    def backward_propagation(self, cache, X, Y):

        m = X.shape[1]
        W1 = self.parameters["W1"]
        W2 = self.parameters["W2"]
        A1 = cache["A1"]
        A2 = cache["A2"]
        dZ2 = A2-Y
        dW2 = np.dot(dZ2, A1.T)/m
        db2 = np.sum(dZ2, axis=1, keepdims=True)/m
        dZ1 = np.dot(W2.T, dZ2)*(1-np.power(A1, 2))
        dW1 = np.dot(dZ1, X.T)/m
        db1 = np.sum(dZ1, axis=1, keepdims=True)/m

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return grads

    def update_parameters(self, grads):
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]

        W1 = W1-self.learning_rate*dW1
        b1 = b1-self.learning_rate*db1
        W2 = W2-self.learning_rate*dW2
        b2 = b2-self.learning_rate*db2

        self.parameters = {"W1": W1,
                           "b1": b1,
                           "W2": W2,
                           "b2": b2}

        return self.parameters

    def fit(self, X, Y, print_cost=False):

        np.random.seed(3)
        n_x = self.layer_sizes(X, Y)[0]
        n_y = self.layer_sizes(X, Y)[1]

        self.parameters = self.intialize_weights(n_x, self.n_h, n_y)

        for i in range(0, self.num_iterations):
            A2, cache = self.forward_propagation(X)

            cost = self.compute_cost(A2, Y)

            grads = self.backward_propagation(cache, X, Y)

            self.parameters = self.update_parameters(grads)

            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        return self.parameters

    def predict(self, X):
        A2, cache = self.forward_propagation(X)
        predictions = np.round(A2)

        return predictions

