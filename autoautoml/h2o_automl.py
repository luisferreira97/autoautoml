from h2o.automl import H2OAutoML
import h2o

class H2O:

    def __init__(self):
        self.tool = "H2O AutoML"
    
    def run_example(self):

        h2o.init()

        # Import a sample binary outcome train/test set into H2O
        train = h2o.import_file("./data/churn-train.csv")
        test = h2o.import_file("./data/churn-test.csv")
        #df = h2o.import_file("./data/churn.csv")
        #train, test = df.split_frame(ratios=[.75])

        # Identify predictors and response
        x = train.columns
        y = "churn_probability"
        x.remove(y)

        # For binary classification, response should be a factor
        #train[y] = train[y].asfactor()
        #test[y] = test[y].asfactor()

        # Run AutoML for 20 base models (limited to 1 hour max runtime by default)
        aml = H2OAutoML(max_runtime_secs=20, seed=1, sort_metric = "mae")
        aml.train(x=x, y=y, training_frame=train)

        # View the AutoML Leaderboard
        lb = aml.leaderboard
        lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)


        # The leader model is stored here
        print(aml.leader.model_performance(test))

        # If you need to generate predictions on a test set, you can make
        # predictions directly on the `"H2OAutoML"` object, or on the leader
        # model object directly

        preds = aml.predict(test)

        # or:
        preds = aml.leader.predict(test)
        print(preds)

        resp = [aml, aml.leader, preds.as_data_frame()]
        
        h2o.shutdown()

        return resp

    def run(self, train_path, test_path, target, task):
        train = h2o.import_file(train_path)
        test = h2o.import_file(test_path)

        x = train.columns
        y = target
        x.remove(y)

        if task == 'class':
            train[y] = train[y].asfactor()
            test[y] = test[y].asfactor()

        aml = H2OAutoML(seed=42, sort_metric = "auto", nfolds=5, exclude_algos=["DeepLearning"])
        aml.train(x=x, y=y, training_frame=train)       
        preds = aml.leader.predict(test)
        print(aml.leader.model_performance(test))
        
        return aml.leader.model_performance(test)

