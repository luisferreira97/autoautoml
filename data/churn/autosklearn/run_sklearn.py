import pandas as pd
from datetime import datetime
import json

data_path = "./data/churn/churn"

target = 'class'

fold1 = pd.read_csv(data_path + "-fold1.csv")
fold2 = pd.read_csv(data_path + "-fold2.csv")
fold3 = pd.read_csv(data_path + "-fold3.csv")
fold4 = pd.read_csv(data_path + "-fold4.csv")
fold5 = pd.read_csv(data_path + "-fold5.csv")
fold6 = pd.read_csv(data_path + "-fold6.csv")
fold7 = pd.read_csv(data_path + "-fold7.csv")
fold8 = pd.read_csv(data_path + "-fold8.csv")
fold9 = pd.read_csv(data_path + "-fold9.csv")
fold10 = pd.read_csv(data_path + "-fold10.csv")

folds = [fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9, fold10]

import sklearn.model_selection
import autosklearn.regression
import autosklearn.classification
import sklearn.metrics

for x in range(0, 10):
    fold_folder = "./data/churn/autosklearn/fold" + str(x+1)
    folds = [fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9, fold10]
    test_df = folds[x]
    X_test = test_df.drop(columns=[target]).to_numpy()
    y_test = test_df[target].to_numpy()

    del folds[x]
    train_df = pd.concat(folds)
    X_train = train_df.drop(columns=[target]).to_numpy()
    y_train = train_df[target].to_numpy()

    automl = autosklearn.classification.AutoSklearnClassifier(
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5}
    )

    start = datetime.now().strftime("%H:%M:%S")
    automl.fit(X_train.copy(), y_train.copy(), metric=autosklearn.metrics.roc_auc)
    automl.refit(X_train.copy(), y_train.copy())
    end = datetime.now().strftime("%H:%M:%S")

    predictions = automl.predict(X_test)

    perf = {}
    perf["start"] = start
    perf["end"] = end
    perf["statistics"] = automl.sprint_statistics()
    perf["pred_score"] = sklearn.metrics.roc_auc_score(y_test, predictions)

    perf = json.dumps(perf)
    f = open(fold_folder + "/perf.json","w")
    f.write(perf)
    f.close()

    from joblib import dump, load
    dump(automl, fold_folder + '/model.joblib')