library(rminer)
library(jsonlite)

data_path = "./data/dmft/dmft"
label_column = 'Prevention'
metric="macroF1"
task = "class"

fold1 = read.csv(paste(data_path, "-fold1.csv", sep = ""), sep=",", header = T)
fold2 = read.csv(paste(data_path, "-fold2.csv", sep = ""), sep=",", header = T)
fold3 = read.csv(paste(data_path, "-fold3.csv", sep = ""), sep=",", header = T)
fold4 = read.csv(paste(data_path, "-fold4.csv", sep = ""), sep=",", header = T)
fold5 = read.csv(paste(data_path, "-fold5.csv", sep = ""), sep=",", header = T)
fold6 = read.csv(paste(data_path, "-fold6.csv", sep = ""), sep=",", header = T)
fold7 = read.csv(paste(data_path, "-fold7.csv", sep = ""), sep=",", header = T)
fold8 = read.csv(paste(data_path, "-fold8.csv", sep = ""), sep=",", header = T)
fold9 = read.csv(paste(data_path, "-fold9.csv", sep = ""), sep=",", header = T)
fold10 = read.csv(paste(data_path, "-fold10.csv", sep = ""), sep=",", header = T)

for (x in 0:9) {
    fold_folder = paste("./data/dmft/rminer-xgboost/fold", as.character(x+1), sep="")
    folds = list(fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9, fold10)
    test_df = folds[[(x+1)]]
    folds = folds[-(x+1)]
    train_df <- do.call(rbind, folds)

    inputs=ncol(train_df)-1

    train_df$Prevention = as.factor(train_df$Prevention)
    test_df$Prevention = as.factor(test_df$Prevention)

    sm=mparheuristic(model="xgboost",n="heuristic10",task=task, inputs=inputs)
    method=c("kfold",5,123)
    search=list(search=sm,smethod="grid",method=method,metric=metric,convex=0)

    M=fit(Prevention~.,data=train_df,model="xgboost",search=search,fdebug=TRUE)

    P=predict(M,test_df)

    results = paste('{"time":',
                    M@time,
                    ',"validation_metric":',
                    M@error,
                    ',"test_metric":',
                    round(mmetric(test_df$Prevention,P,metric=metric),2),
                    '}',
                    sep = "")

    write(results, paste(fold_folder, "/perf.json", sep= ""))
    save(M, file = paste(fold_folder, "/model.RData", sep= ""))
}
