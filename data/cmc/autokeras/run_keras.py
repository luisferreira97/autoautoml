import json
import os
from datetime import datetime
from shutil import rmtree

import autokeras as ak
import pandas as pd
import sklearn.metrics

data_path = "./data/cmc/cmc"

target = "Contraceptive_method_used"

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


for x in range(9, 10):
    fold_folder = "./data/cmc/autokeras/fold" + str(x + 1)
    folds = [fold1, fold2, fold3, fold4, fold5,
             fold6, fold7, fold8, fold9, fold10]
    test_df = folds[x]
    X_test = test_df.drop(columns=[target]).to_numpy()
    y_test = test_df[target].to_numpy()

    del folds[x]
    train_df = pd.concat(folds)
    X_train = train_df.drop(columns=[target]).to_numpy()
    y_train = train_df[target].to_numpy()

    classifier = ak.StructuredDataClassifier(
        objective="val_loss", directory=fold_folder, project_name="results"
    )

    start = datetime.now().strftime("%H:%M:%S")
    classifier.fit(x=X_train, y=y_train, validation_split=0.25)
    end = datetime.now().strftime("%H:%M:%S")

    try:
        classifier.evaluate(x=X_test, y=y_test)
    except Exception as e:
        error = str(e)

    best_trial = error[error.find("trial"): error.find("/checkpoints")]

    trials = os.listdir(fold_folder + "/results")

    for trial in trials:
        if trial == best_trial:
            checkpoints = os.listdir(
                fold_folder + "/results/" + trial + "/checkpoints")
            checkpoints = [int(checkpoint[6:]) for checkpoint in checkpoints]
            checkpoints.sort()
            if checkpoints[0] == 0:
                del checkpoints[0]
            best = min(checkpoints)
            os.rename(
                fold_folder + "/results/" + trial + "/checkpoints/epoch_0",
                fold_folder
                + "/results/"
                + trial
                + "/checkpoints/epoch_"
                + str(best - 1),
            )
        elif "trial" in trial:
            rmtree(fold_folder + "/results/" + trial)

    preds = classifier.predict(x=X_test)

    perf = {}
    perf["start"] = start
    perf["end"] = end
    perf["test_score"] = sklearn.metrics.f1_score(
        y_test, preds.reshape((len(preds),)), average="macro"
    )

    perf = json.dumps(perf)
    f = open(fold_folder + "/perf.json", "w")
    f.write(perf)
    f.close()

    model = classifier.export_model()

    try:
        model.save(fold_folder + "/model_autokeras", save_format="tf")
    except:
        model.save(fold_folder + "/model_autokeras.h5")
