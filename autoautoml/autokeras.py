import autokeras as ak

class AUTOKERAS():

    def __init__(self):
        self.algo = "AUTOKERAS"

    def run_example(self):

        # Initialize the classifier.
        clf = ak.StructuredDataClassifier(max_trials=30)
        # x is the path to the csv file. y is the column name of the column to predict.
        clf.fit(x='PATH_TO/train.csv', y='survived')
        # Evaluate the accuracy of the found model.
        #print('Accuracy: {accuracy}'.format(accuracy=clf.evaluate(x='PATH_TO/eval.csv', y='survived')))