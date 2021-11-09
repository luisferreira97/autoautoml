import json
from datetime import datetime

import h2o
import pandas as pd
from h2o.automl import H2OAutoML

data_path = "./data/plasma/plasma"

target = "RETPLASMA"

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
    h2o.init()

    fold_folder = "./data/plasma/h2o-no-DL/fold" + str(x + 1)
    folds = [fold1, fold2, fold3, fold4, fold5,
             fold6, fold7, fold8, fold9, fold10]
    test_df = folds[x]
    test = h2o.H2OFrame(test_df)

    del folds[x]
    train_df = pd.concat(folds)
    train = h2o.H2OFrame(train_df)

    x = train.columns
    y = target
    x.remove(y)

    aml = H2OAutoML(
        seed=42,
        sort_metric="mae",
        nfolds=5,
        exclude_algos=["DeepLearning"],
        max_runtime_secs=3600,
    )

    start = datetime.now().strftime("%H:%M:%S")
    aml.train(x=x, y=y, training_frame=train)
    end = datetime.now().strftime("%H:%M:%S")

    lb = aml.leaderboard.as_data_frame()
    lb.to_csv(fold_folder + "/leaderboard.csv", index=False)

    perf = aml.leader.model_performance(test)

    perf = aml.training_info
    perf["start"] = start
    perf["end"] = end
    perf["metric"] = aml.leader.model_performance(test).mae()

    perf = json.dumps(perf)
    f = open(fold_folder + "/perf.json", "w")
    f.write(perf)
    f.close()

    my_local_model = h2o.download_model(aml.leader, path=fold_folder)

    h2o.shutdown()

    import time

    time.sleep(5)
