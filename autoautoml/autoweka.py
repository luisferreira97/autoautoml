import pyautoweka
#load the data:
import numpy as np
import random

class AUTOWEKA():

    def __init__(self):
        self.algo = "TPOT"

    def run_example(self):

        #Create an experiment
        experiment = pyautoweka.ClassificationExperiment(tuner_timeout=360)

        X = np.loadtxt("iris.data", delimiter=",", usecols=range(4))
        y = np.loadtxt("iris.data", delimiter=",", usecols=[4], dtype="object")

        #shuffle the data:
        indices = range(len(X))
        random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        #split into train and test set:
        X_train = X[0:100]
        y_train = y[0:100]

        X_test = X[100:]
        y_test = y[100:]

        #now we can fit a model:
        experiment.fit(X_train, y_train)

        #and predict the labels of the held out data:
        y_predict = experiment.predict(X_test)

        #Let's check what accuracy we get:
        num_correct = sum([1 for predicted, correct in zip(y_predict, y_test) if predicted == correct])
        print("Accuracy: %f" % (float(num_correct) / len(y_test)))
