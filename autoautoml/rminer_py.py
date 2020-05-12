import rpy2.robjects.packages as rpackages
from rpy2 import robjects
base = rpackages.importr('base')
utils = rpackages.importr('utils')
rminer = rpackages.importr('rminer')

train = robjects.r("""train=read.csv("/home/lferreira/autoautoml/data/churn-train.csv", sep=",", header = T)""")
train = robjects.r("""train=train[,-14]""")

test = robjects.r("""test=read.csv("/home/lferreira/autoautoml/data/churn-test.csv", sep=",", header = T)""")
train = robjects.r("""test=train[,-14]""")

inputs = robjects.r("""inputs=ncol(train)""")
metric = robjects.r("""metric='MAE'""")
sm = robjects.r("""sm=mparheuristic(model="automl3",n=NA,task="reg", inputs= inputs)""")
method = robjects.r("""method=c("kfold",3,123)""")
search = robjects.r("""search=list(search=sm,smethod="auto",method=method,metric=metric,convex=0)""")

M = robjects.r("""M=fit(churn_probability~.,data=train,model="auto",search=search,fdebug=TRUE)""")
P = robjects.r("""P=predict(M,test)""")

# show leaderboard:
robjects.r("""cat("> leaderboard models:",M@mpar$LB$model,"\n")""")
robjects.r("""cat(">  validation values:",round(M@mpar$LB$eval,4),"\n")""")
robjects.r("""cat("best model is:",M@model,"\n")""")
robjects.r("""cat(metric,"=",round(mmetric(test$churn_probability,P,metric=metric),2),"\n")""")
