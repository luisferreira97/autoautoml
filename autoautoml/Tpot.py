from tpot import TPOTRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class TPOT():
    
    def __init__(self):
        self.algo = "TPOT"

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

        tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42, scoring='neg_mean_absolute_error', cv=5)
        tpot.fit(X_train, y_train)
        print(tpot.score(X_test, y_test))
        tpot.export('tpot_iris_pipeline.py')

        return tpot.score(X_test, y_test)

    def run(self, train_path, test_path, target, task):
        train = pd.read_csv(train_path)
        X_train = train.drop(columns=[target]).to_numpy()
        y_train = train[target].to_numpy()

        test = pd.read_csv(test_path)
        X_test = test.drop(columns=[target]).to_numpy()
        y_test = test[target].to_numpy()

        if task == "class":
            tpot = TPOTClassifier(max_time_mins=60, verbosity=3, random_state=42, scoring='roc_auc', cv=5, n_jobs=-1, early_stop=3)
        elif task == "reg":
            tpot = TPOTRegressor(max_time_mins=60, verbosity=3, random_state=42, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, early_stop=3)
        
        tpot.fit(X_train, y_train)
        print(tpot.score(X_test, y_test))
        tpot.export('/home/lferreira/autoautoml/data/churn/tpot/plasma.py')

        return tpot.score(X_test, y_test)