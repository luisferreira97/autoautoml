import autogluon as ag
from autogluon import TabularPrediction as task

class AUTOGLUON():
    
    def __init__(self):
        self.algo = "Auto-gluon"

    def run_example(self):
        
        train_data = task.Dataset(file_path='./data/churn-train.csv')
        train_data = train_data.head(500) # subsample 500 data points for faster demo
        print(train_data.head())
        label_column = 'churn_probability'
        print("Summary of class variable: \n", train_data[label_column].describe())
        dir = 'agModels-predictClass' # specifies folder where to store trained models
        predictor = task.fit(train_data=train_data, label=label_column, eval_metric="mean_absolute_error")
        test_data = task.Dataset(file_path='./data/churn-test.csv')
        y_test = test_data[label_column]  # values to predict
        test_data_nolab = test_data.drop(labels=[label_column],axis=1) # delete label column to prove we're not cheating
        print(test_data_nolab.head())
        #predictor = task.load(dir) # unnecessary, just demonstrates how to load previously-trained predictor from file

        y_pred = predictor.predict(test_data_nolab)
        print("Predictions:  ", y_pred)
        perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

        print("MAE: " + perf)

        return perf

    def run(self, train_path, test_path, target, task):
        train_data = task.Dataset(file_path=train_path)

        predictor = task.fit(train_data=train_data, label=label_column, eval_metric="f1_macro", num_bagging_folds=5)
        
        test_data = task.Dataset(file_path=test_path)
        y_test = test_data[target]

        y_pred = predictor.predict(test_data)
        return predictor.evaluate_predictions(y_true=y_test.to_numpy(), y_pred=y_pred, auxiliary_metrics=True)