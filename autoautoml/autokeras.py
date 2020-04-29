import autokeras as ak

class AUTOKERAS():

    def __init__(self):
        self.algo = "AUTOKERAS"

    def run_example(self):

        # Initialize the classifier.
        clf = ak.StructuredDataRegressor(max_trials=30, loss="mean_absolute_error")
        # x is the path to the csv file. y is the column name of the column to predict.

        clf.fit(x='./data/churn2.csv', y='churn_probability')
        
        # Evaluate the accuracy of the found model.
        print('Accuracy: {accuracy}'.format(accuracy=clf.evaluate(x='PATH_TO/eval.csv', y='survived')))