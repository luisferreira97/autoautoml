import pandas as pd
from datetime import datetime
import json

data_path = "./data/dmft/dmft"

target = 'Prevention'

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

from tpot import TPOTRegressor
from tpot import TPOTClassifier

for x in range(0, 10):
    fold_folder = "./data/dmft/tpot/fold" + str(x+1)
    folds = [fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9, fold10]
    test_df = folds[x]
    X_test = test_df.drop(columns=[target]).to_numpy()
    y_test = test_df[target].to_numpy()

    del folds[x]
    train_df = pd.concat(folds)
    X_train = train_df.drop(columns=[target]).to_numpy()
    y_train = train_df[target].to_numpy()

    tpot = TPOTClassifier(max_time_mins=60, verbosity=3, random_state=42, scoring='f1_macro', cv=5, n_jobs=-1, early_stop=3)

    from datetime import datetime
    start = datetime.now().strftime("%H:%M:%S")
    tpot.fit(X_train, y_train)
    end = datetime.now().strftime("%H:%M:%S")

    tpot.export(fold_folder + "/pipeline.py")
    perf = {
        "score": tpot.score(X_test, y_test),
        "start": start,
        "end": end
    }
    perf = json.dumps(perf)
    f = open(fold_folder + "/perf.json","w")
    f.write(perf)
    f.close()