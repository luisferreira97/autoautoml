import json
from datetime import datetime

import pandas as pd
from autogluon import TabularPrediction as task

data_path = "./data/cholesterol/cholesterol"

label_column = "chol"

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


leaderboards = []
metrics = []

for x in range(0, 10):
    fold_folder = "./data/cholesterol/autogluon/fold" + str(x + 1)
    folds = [fold1, fold2, fold3, fold4, fold5,
             fold6, fold7, fold8, fold9, fold10]
    test_df = folds[x]
    del folds[x]
    train_df = pd.concat(folds)
    train_data = task.Dataset(train_df)
    start = datetime.now().strftime("%H:%M:%S")
    predictor = task.fit(
        train_data=train_data,
        label=label_column,
        eval_metric="mean_absolute_error",
        num_bagging_folds=5,
        output_directory=fold_folder,
    )
    end = datetime.now().strftime("%H:%M:%S")
    predictor.leaderboard().to_csv(fold_folder + "/leaderboard.csv")
    test_data = task.Dataset(test_df)
    y_test = test_data[label_column]
    y_pred = predictor.predict(test_data)
    perf = predictor.evaluate_predictions(
        y_true=y_test.to_numpy(), y_pred=y_pred, auxiliary_metrics=True
    )
    perf = dict(perf)
    perf["start"] = start
    perf["end"] = end
    perf = json.dumps(perf)
    f = open(fold_folder + "/perf.json", "w")
    f.write(perf)
    f.close()
