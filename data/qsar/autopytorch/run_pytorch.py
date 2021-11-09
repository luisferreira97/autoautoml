import json
import sys
from datetime import datetime

import pandas as pd
import sklearn.metrics
import torch
from autoPyTorch import AutoNetClassification, AutoNetRegression

data_path = "./data/qsar/qsar"

target = "Class"

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


for x in range(0, 10):
    fold_folder = "./data/qsar/autopytorch/fold" + str(x + 1)
    folds = [fold1, fold2, fold3, fold4, fold5,
             fold6, fold7, fold8, fold9, fold10]
    test_df = folds[x]
    X_test = test_df.drop(columns=[target]).to_numpy()
    y_test = test_df[target].to_numpy()

    del folds[x]
    train_df = pd.concat(folds)
    X_train = train_df.drop(columns=[target]).to_numpy()
    y_train = train_df[target].to_numpy()

    autonet = AutoNetClassification(
        "tiny_cs",  # config preset
        log_level="info",
        budget_type="time",
        max_runtime=300,
        min_budget=5,
        max_budget=100,
    )

    start = datetime.now().strftime("%H:%M:%S")
    perf = autonet.fit(
        X_train=X_train,
        Y_train=y_train,
        validation_split=0.25,
        optimize_metric="auc_metric",
    )
    end = datetime.now().strftime("%H:%M:%S")

    preds = autonet.predict(X=X_test)

    perf["start"] = start
    perf["end"] = end
    perf["test_score"] = sklearn.metrics.roc_auc_score(y_test, preds)

    perf = json.dumps(perf)
    f = open(fold_folder + "/perf.json", "w")
    f.write(perf)
    f.close()

    torch.save(autonet.get_pytorch_model(), fold_folder + "/model")
