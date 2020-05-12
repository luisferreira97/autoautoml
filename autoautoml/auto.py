# Auto-keras
import autokeras as ak

# Auto-gluon
import autogluon as ag
from autogluon import TabularPrediction as task

# Auto-sklearn
import autosklearn.regression
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# H2O AutoML
from h2o.automl import H2OAutoML
import h2o

# TPOT
from tpot import TPOTRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Autoautoml():
    def __init__(self):
        self.algo = "Auto-Auto"

    def train(self, train_path, test_path, algos):
        a = 1+1
        return a

    def run_example(self, train_path, test_path, target):

        metrics = {}

        train = pd.read_csv("./data/churn-train.csv") 

        # Auto-keras
        regressor = ak.StructuredDataRegressor(max_trials=10, loss="mean_absolute_error")
        regressor.fit(x='./data/churn-train.csv', y=target)
        metrics["auto-keras"] = regressor.evaluate(x='./data/churn-test.csv', y=target)[0]

        # Auto-gluon
        train_data = task.Dataset(file_path='./data/churn-train.csv')
        label_column = target
        predictor = task.fit(train_data=train_data, label=label_column, eval_metric="mean_absolute_error")
        test_data = task.Dataset(file_path='./data/churn-test.csv')
        y_test = test_data[label_column]  # values to predict
        test_data_nolab = test_data.drop(labels=[label_column],axis=1) # delete label column to prove we're not cheating
        y_pred = predictor.predict(test_data_nolab)
        metrics["auto-gluon"] = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)["mean_absolute_error"]

        # Auto-sklearn
        categorical_feature_mask = train.dtypes==object
        categorical_cols = train.columns[categorical_feature_mask].tolist()
        le = LabelEncoder()
        train[categorical_cols] = train[categorical_cols].apply(lambda col: le.fit_transform(col))
        X_train = train.drop(columns=[target]).to_numpy()
        y_train = train[target].to_numpy()
        test = pd.read_csv("./data/churn-test.csv")
        test[categorical_cols] = test[categorical_cols].apply(lambda col: le.fit_transform(col))
        X_test = test.drop(columns=[target]).to_numpy()
        y_test = test[target].to_numpy()
        automl = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=120,
            per_run_time_limit=30,
            resampling_strategy='cv',
            resampling_strategy_arguments={'folds': 5},
        )
        automl.fit(X_train.copy(), y_train.copy(), metric=autosklearn.metrics.mean_absolute_error)
        automl.refit(X_train.copy(), y_train.copy())
        predictions = automl.predict(X_test)
        metrics["auto-sklearn"] = sklearn.metrics.mean_absolute_error(y_test, predictions)

        # H2O AutoML
        h2o.init()

        train = h2o.import_file("./data/churn-train.csv")
        test = h2o.import_file("./data/churn-test.csv")
        x = train.columns
        y = target
        x.remove(y)
        aml = H2OAutoML(max_runtime_secs=20, seed=1, sort_metric = "mae")
        aml.train(x=x, y=y, training_frame=train)
        metrics["h2o-automl"] = aml.leader.model_performance(test).mae()
        
        h2o.shutdown()

        # TPOT
        tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42, scoring='neg_mean_absolute_error', cv=5)
        tpot.fit(X_train, y_train)
        metrics["tpot"] = -tpot.score(X_test, y_test)

        best_metric = float("inf")
        best_model = "MODEL"
        for metric in metrics:
            if metrics[metric] < best_metric:
                best_metric = metrics[metric]
                best_model = metric

        print("THE BEST AUTOML TOOL IS " + str(best_model) + ", WITH A MAE OF " + str(best_metric) + " ACHIEVED BY THE BEST MODEL.")
        
        return metrics
        