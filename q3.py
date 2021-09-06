from models.bpn import BPN
import numpy as np
X = [
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]],#0
    [[0, 1, 0], [0, 1, 0], [0, 1, 0]],#1
    [[1, 1, 1], [0, 1, 0], [1, 1, 1]],#2
    [[1, 1, 1], [0, 1, 1], [1, 1, 1]],#3
    [[1, 0, 1], [1, 1, 1], [0, 0, 1]],#4
    [[1, 1, 1], [1, 1, 0], [1, 1, 1]],#5
    [[1, 0, 0], [1, 1, 1], [1, 1, 1]],#6
    [[1, 1, 1], [0, 0, 1], [0, 0, 1]],#7
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],#8
    [[1, 1, 1], [1, 1, 1], [0, 0, 1]]#9
]

Y = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0],
     [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1]]


def decodeDigits(result):
    print("Predicted Vector : ", result.T)
    bn = "".join([str(int(x)) for x in (list(result.T[0]))])
    print("Predicted Digit : ", int(bn, 2))


def printDigit(X):
    print("Digit : ")
    for x in range(len(X)):
        for y in range(len(X[0])):
            if(X[x][y] == 0):
                print("O", end=" ")
            else:
                print("X", end=" ")
        print()


if __name__ == '__main__':
    X_train = np.array([np.array(i).flatten() for i in np.asarray(X)]).T
    Y_train = np.array(Y).T
    classifier = BPN(hidden_neurons=6)
    classifier.fit(X_train, Y_train)
    print("Parameters :\n" , classifier.parameters)
    print("\nTesting")
    test1 = np.array(X[1]).flatten().reshape(9, 1)  # 1 shape
    printDigit(X[1])
    decodeDigits(classifier.predict(test1))

    test2 = np.array(X[9]).flatten().reshape(9, 1)  # 9 shape
    printDigit(X[9])
    decodeDigits(classifier.predict(test2))
