library(rminer)

train_path = "/home/lferreira/autoautoml/data/mfeat/mfeat-train.csv"
test_path = "/home/lferreira/autoautoml/data/mfeat/mfeat-test.csv"
target = "class"
task = "class"

train=read.csv(train_path, sep=",", header = T)
test=read.csv(test_path, sep=",", header = T)

train[,c(target)] = as.factor(train[,c(target)])
test[,c(target)] = as.factor(test[,c(target)])

inputs=ncol(train)-1
metric="macroF1"
sm=mparheuristic(model="automl3",n=NA,task=task, inputs= inputs)
method=c("kfold",5,123)
search=list(search=sm,smethod="auto",method=method,metric=metric,convex=0)

M=fit(class~.,data=train,model="auto",search=search,fdebug=TRUE)

P=predict(M,test)

# show leaderboard:
cat("> time:",M@time,"\n")
cat("> leaderboard models:",M@mpar$LB$model,"\n")
cat(">  validation values:",round(M@mpar$LB$eval,4),"\n")
cat("best model is:",M@model,"\n")
cat(metric,"=",round(mmetric(test$class,P,metric=metric),2),"\n")

save(M, file = "model.RData")

