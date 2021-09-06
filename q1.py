from models.preceptron import Preceptron
def dectobin(num):
    x = []
    while num > 0:
        x = [num % 2] + x
        num = num//2
    return x


def outputvector():
    output = []
    for i in range(65, 91):
        output.append(dectobin(i))
    return output


def decodeDigits(result):
    print("\nPredicted Vector : ", result)
    bn = "".join([str(int(x)) for x in (list(result))])
    print("Predicted Alphabet : ", chr(int(bn, 2)))
    print()


def printDigit(X):
    print("Digit : ",end="")
    for x in range(len(X)):
        if(x % 3 == 0):
            print()
        if(X[x] == 0):
            print("O", end=" ")
        else:
            print("X", end=" ")
        



dataset = {
    "X": [[0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
     [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
      [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1], 
      [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
      [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1], 
      [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0], 
      [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1], 
      [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
      [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], 
      [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], 
      [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1], 
      [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
      [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], 
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
      [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0], 
      [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
      [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1], 
      [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1], 
      [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1], 
      [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
      [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
      [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
      [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
      [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
      [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
      [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]],
    "y": outputvector()
}

if __name__ == "__main__":

    classifier = Preceptron()
    parameters = classifier.fit(dataset["X"], dataset["y"])
    
    print("Parameters :\n" , parameters)
    print("\nTesting")
    test1 = dataset["X"][0]  # A shape
    printDigit(test1)
    decodeDigits(classifier.predict(test1))

    test2 = dataset["X"][9]  # J shape
    printDigit(test2)
    decodeDigits(classifier.predict(test2))
