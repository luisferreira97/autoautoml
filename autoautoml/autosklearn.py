import autosklearn.regression
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class AUTOSKLEARN():
    
    def __init__(self):
        self.algo = "Auto-sklearn"

    def run_example(self):

        train = pd.read_csv("./data/churn-train.csv") 
        #dummy_train = pd.get_dummies(train[categorical_cols])  
        categorical_feature_mask = train.dtypes==object
        categorical_cols = train.columns[categorical_feature_mask].tolist()
        le = LabelEncoder()
        #le.fit(train[categorical_cols])
        #le.transform(train[categorical_cols])
        train[categorical_cols] = train[categorical_cols].apply(lambda col: le.fit_transform(col))
        # numpy
        X_train = train.drop(columns=['churn_probability']).to_numpy()
        y_train = train["churn_probability"].to_numpy()
        

        test = pd.read_csv("./data/churn-test.csv")
        #dummy_new = pd.get_dummies(test[categorical_cols])
        test[categorical_cols] = test[categorical_cols].apply(lambda col: le.fit_transform(col))
        X_test = test.drop(columns=['churn_probability']).to_numpy()
        y_test = test["churn_probability"].to_numpy()

        automl = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=120,
            per_run_time_limit=30,
            resampling_strategy='cv',
            resampling_strategy_arguments={'folds': 5},
        )

        automl.fit(X_train.copy(), y_train.copy(), metric=autosklearn.metrics.mean_absolute_error)
        automl.refit(X_train.copy(), y_train.copy())

        print(automl.show_models())
        predictions = automl.predict(X_test)
        print("MAE score:", sklearn.metrics.mean_absolute_error(y_test, predictions))

    def run(self, train_path, test_path, target, task):
        train = pd.read_csv(train_path) 
        X_train = train.drop(columns=[target]).to_numpy()
        y_train = train[target].to_numpy()

        test = pd.read_csv(test_path)
        X_test = test.drop(columns=[target]).to_numpy()
        y_test = test[target].to_numpy()

        if task == "reg":
            automl = autosklearn.regression.AutoSklearnRegressor(
                resampling_strategy='cv',
                resampling_strategy_arguments={'folds': 5}
            )
        elif task == "class":
            automl = autosklearn.classification.AutoSklearnClassifier(
                resampling_strategy='cv',
                resampling_strategy_arguments={'folds': 5}
            )
        
        automl.fit(X_train.copy(), y_train.copy())
        automl.refit(X_train.copy(), y_train.copy())

        print(automl.sprint_statistics())
        predictions = automl.predict(X_test)
        
        return predictions