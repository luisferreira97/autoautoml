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

from sklearn.preprocessing import LabelEncoder
import numpy as np


class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)