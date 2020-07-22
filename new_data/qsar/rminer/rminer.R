library(rminer)
library(jsonlite)

data_path = "/home/lferreira/autoautoml/new_data/qsar/qsar"
label_column = 'Class'
metric="AUC"
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
    fold_folder = paste("/home/lferreira/autoautoml/new_data/qsar/rminer/fold", as.character(x+1), sep="")
    folds = list(fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9, fold10)
    test_df = folds[[(x+1)]]
    folds = folds[-(x+1)]
    train_df <- do.call(rbind, folds)

    inputs=ncol(train_df)-1

    train_df$Class = as.factor(train_df$Class)
    test_df$Class = as.factor(test_df$Class)

    sm=mparheuristic(model="automl3",n=NA,task=task, inputs=inputs)
    method=c("kfold",5,123)
    search=list(search=sm,smethod="auto",method=method,metric=metric,convex=0)

    M=fit(Class~.,data=train_df,model="auto",search=search,fdebug=TRUE)

    P=predict(M,test_df)

    test_metric = round(mmetric(test_df$Class,P,metric=metric),2)
    best_model = M@model
    best_model_index = match(best_model,M@mpar$LB$model)
    validation_metric = round(M@mpar$LB$eval,4)[best_model_index]

    results = paste('{"time":',
                    M@time, 
                    ',"best_model":"',
                    best_model,
                    '","validation_metric":',
                    validation_metric,
                    ',"test_metric":',
                    test_metric,
                    '}',
                    sep = "")

    write(results, paste(fold_folder, "/perf.json", sep= ""))
    save(M, file = paste(fold_folder, "/model.RData", sep= ""))
}