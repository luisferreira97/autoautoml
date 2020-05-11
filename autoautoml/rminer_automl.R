#if (!("package:rminer" %in% search())) { install.packages("rminer") }

library(rminer)
train=read.csv("/home/lferreira/autoautoml/data/churn-train.csv", sep=",", header = T)
test=read.csv("/home/lferreira/autoautoml/data/churn-test.csv", sep=",", header = T)

inputs=ncol(train)-1
metric="MAE"
sm=mparheuristic(model="automl3",n=NA,task="reg", inputs= inputs)
method=c("kfold",3,123)
search=list(search=sm,smethod="auto",method=method,metric=metric,convex=0)

M=fit(churn_probability~.,data=train,model="auto",search=search,fdebug=TRUE)
P=predict(M,test)

# show leaderboard:
cat("> leaderboard models:",M@mpar$LB$model,"\n")
cat(">  validation values:",round(M@mpar$LB$eval,4),"\n")
cat("best model is:",M@model,"\n")
cat(metric,"=",round(mmetric(test$churn_probability,P,metric=metric),2),"\n")