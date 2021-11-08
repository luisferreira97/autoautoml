import pandas as pd

paths = [
    "./data/cholesterol/cholesterol",
    "./data/churn/churn",
    "./data/cloud/cloud",
    "./data/cmc/cmc",
    "./data/credit/credit",
    "./data/diabetes/diabetes",
    "./data/dmft/dmft",
    "./data/liver-disorders/liver-disorders",
    "./data/mfeat/mfeat",
    "./data/plasma/plasma",
    "./data/qsar/qsar",
    "./data/vehicle/vehicle"
]

folds_paths = [
    "./data/cholesterol/transmogrifai/fold",
    "./data/churn/transmogrifai/fold",
    "./data/cloud/transmogrifai/fold",
    "./data/cmc/transmogrifai/fold",
    "./data/credit/transmogrifai/fold",
    "./data/diabetes/transmogrifai/fold",
    "./data/dmft/transmogrifai/fold",
    "./data/liver-disorders/transmogrifai/fold",
    "./data/mfeat/transmogrifai/fold",
    "./data/plasma/transmogrifai/fold",
    "./data/qsar/transmogrifai/fold",
    "./data/vehicle/transmogrifai/fold"
]


for x in range(0, 12):
    path = paths[x]

    fold1 = pd.read_csv(path + "-fold1.csv")
    fold2 = pd.read_csv(path + "-fold2.csv")
    fold3 = pd.read_csv(path + "-fold3.csv")
    fold4 = pd.read_csv(path + "-fold4.csv")
    fold5 = pd.read_csv(path + "-fold5.csv")
    fold6 = pd.read_csv(path + "-fold6.csv")
    fold7 = pd.read_csv(path + "-fold7.csv")
    fold8 = pd.read_csv(path + "-fold8.csv")
    fold9 = pd.read_csv(path + "-fold9.csv")
    fold10 = pd.read_csv(path + "-fold10.csv")

    for y in range(0, 10):
        fold_folder = folds_paths[x] + str(y + 1)
        folds = [fold1, fold2, fold3, fold4, fold5,
                 fold6, fold7, fold8, fold9, fold10]
        test_df = folds[y]
        del folds[y]
        train_df = pd.concat(folds)
        train_df.to_csv(fold_folder + "/train.csv", index=False)
        test_df.to_csv(fold_folder + "/test.csv", index=False)
