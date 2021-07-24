import pandas as pd
paths = [
    "/home/lferreira/autoautoml/new_data/cholesterol/cholesterol",
    "/home/lferreira/autoautoml/new_data/churn/churn",
    "/home/lferreira/autoautoml/new_data/cloud/cloud",
    "/home/lferreira/autoautoml/new_data/cmc/cmc",
    "/home/lferreira/autoautoml/new_data/credit/credit",
    "/home/lferreira/autoautoml/new_data/diabetes/diabetes",
    "/home/lferreira/autoautoml/new_data/dmft/dmft",
    "/home/lferreira/autoautoml/new_data/liver-disorders/liver-disorders",
    "/home/lferreira/autoautoml/new_data/mfeat/mfeat",
    "/home/lferreira/autoautoml/new_data/plasma/plasma",
    "/home/lferreira/autoautoml/new_data/qsar/qsar",
    "/home/lferreira/autoautoml/new_data/vehicle/vehicle"
]

folds_paths = [
    "/home/lferreira/autoautoml/new_data/cholesterol/transmogrifai/fold",
    "/home/lferreira/autoautoml/new_data/churn/transmogrifai/fold",
    "/home/lferreira/autoautoml/new_data/cloud/transmogrifai/fold",
    "/home/lferreira/autoautoml/new_data/cmc/transmogrifai/fold",
    "/home/lferreira/autoautoml/new_data/credit/transmogrifai/fold",
    "/home/lferreira/autoautoml/new_data/diabetes/transmogrifai/fold",
    "/home/lferreira/autoautoml/new_data/dmft/transmogrifai/fold",
    "/home/lferreira/autoautoml/new_data/liver-disorders/transmogrifai/fold",
    "/home/lferreira/autoautoml/new_data/mfeat/transmogrifai/fold",
    "/home/lferreira/autoautoml/new_data/plasma/transmogrifai/fold",
    "/home/lferreira/autoautoml/new_data/qsar/transmogrifai/fold",
    "/home/lferreira/autoautoml/new_data/vehicle/transmogrifai/fold"
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
