paths = [
    "/home/lferreira/autoautoml/new_data/cholesterol/cholesterol.csv",
    "/home/lferreira/autoautoml/new_data/churn/churn.csv",
    "/home/lferreira/autoautoml/new_data/cloud/cloud.csv",
    "/home/lferreira/autoautoml/new_data/cmc/cmc.csv",
    "/home/lferreira/autoautoml/new_data/credit/credit.csv",
    "/home/lferreira/autoautoml/new_data/diabetes/diabetes.csv",
    "/home/lferreira/autoautoml/new_data/dmft/dmft.csv",
    "/home/lferreira/autoautoml/new_data/liver-disorders/liver-disorders.csv",
    "/home/lferreira/autoautoml/new_data/mfeat/mfeat.csv",
    "/home/lferreira/autoautoml/new_data/plasma/plasma.csv",
    "/home/lferreira/autoautoml/new_data/qsar/qsar.csv",
    "/home/lferreira/autoautoml/new_data/vehicle/vehicle.csv"
]

import pandas as pd 

for path in paths:
    df = pd.read_csv(path)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    kf.split(df)

    indexes = []

    for x in kf.split(df):
        indexes.append(list(x[1]))

    folds = {
        "1": None,
        "2": None,
        "3": None,
        "4": None,
        "5": None,
        "6": None,
        "7": None,
        "8": None,
        "9": None,
        "10": None,
    }

    for x in range(1,11):
        folds[str(x)] = df[df.index.isin(indexes[x-1])]

    folds["1"].to_csv(path[0:len(path)-4] + "-fold1.csv", index=False)
    folds["2"].to_csv(path[0:len(path)-4] + "-fold2.csv", index=False)
    folds["3"].to_csv(path[0:len(path)-4] + "-fold3.csv", index=False)
    folds["4"].to_csv(path[0:len(path)-4] + "-fold4.csv", index=False)
    folds["5"].to_csv(path[0:len(path)-4] + "-fold5.csv", index=False)
    folds["6"].to_csv(path[0:len(path)-4] + "-fold6.csv", index=False)
    folds["7"].to_csv(path[0:len(path)-4] + "-fold7.csv", index=False)
    folds["8"].to_csv(path[0:len(path)-4] + "-fold8.csv", index=False)
    folds["9"].to_csv(path[0:len(path)-4] + "-fold9.csv", index=False)
    folds["10"].to_csv(path[0:len(path)-4] + "-fold10.csv", index=False)
