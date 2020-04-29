from h2o.automl import H2OAutoML
import h2o

class H2O:

    def __init__(self):
        self.tool = "H2O AutoML"
    
    def run_example(self):

        h2o.init()

        # Import a sample binary outcome train/test set into H2O
        train = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv")
        test = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv")

        # Identify predictors and response
        x = train.columns
        y = "response"
        x.remove(y)

        # For binary classification, response should be a factor
        train[y] = train[y].asfactor()
        test[y] = test[y].asfactor()

        # Run AutoML for 20 base models (limited to 1 hour max runtime by default)
        aml = H2OAutoML(max_runtime_secs=20, seed=1)
        aml.train(x=x, y=y, training_frame=train)

        # View the AutoML Leaderboard
        lb = aml.leaderboard
        lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)


        # The leader model is stored here
        aml.leader

        # If you need to generate predictions on a test set, you can make
        # predictions directly on the `"H2OAutoML"` object, or on the leader
        # model object directly

        preds = aml.predict(test)

        # or:
        preds = aml.leader.predict(test)
        print(preds)
        
        h2o.shutdown()

        return [aml, aml.leader, preds]

        