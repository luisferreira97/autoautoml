import autokeras as ak

class AUTOKERAS():

    def __init__(self):
        self.algo = "AUTOKERAS"

    def run_example(self):

        # Initialize the classifier.
        regressor = ak.StructuredDataRegressor(max_trials=10, loss="mean_absolute_error")
        # x is the path to the csv file. y is the column name of the column to predict.

        regressor.fit(x='./data/churn-train.csv', y='churn_probability')
        
        # Evaluate the accuracy of the found model.
        print('MAE: {mae}'.format(mae=regressor.evaluate(x='./data/churn-test.csv', y='churn_probability')))