import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv("PATH/TO/DATA/FILE",
                        sep="COLUMN_SEPARATOR", dtype=np.float64)
features = tpot_data.drop("target", axis=1)
training_features, testing_features, training_target, testing_target = train_test_split(
    features, tpot_data["target"], random_state=42
)

# Average CV score on the training set was: 0.8457905982905984
exported_pipeline = make_pipeline(
    StackingEstimator(
        estimator=DecisionTreeClassifier(
            criterion="entropy", max_depth=1, min_samples_leaf=18, min_samples_split=2
        )
    ),
    StackingEstimator(estimator=BernoulliNB(alpha=0.1, fit_prior=True)),
    RandomForestClassifier(
        bootstrap=True,
        criterion="entropy",
        max_features=0.8500000000000001,
        min_samples_leaf=17,
        min_samples_split=6,
        n_estimators=100,
    ),
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, "random_state", 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
