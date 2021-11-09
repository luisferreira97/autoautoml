from copy import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator, ZeroCount
from tpot.export_utils import set_param_recursive
from xgboost import XGBClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv("PATH/TO/DATA/FILE",
                        sep="COLUMN_SEPARATOR", dtype=np.float64)
features = tpot_data.drop("target", axis=1)
training_features, testing_features, training_target, testing_target = train_test_split(
    features, tpot_data["target"], random_state=42
)

# Average CV score on the training set was: 0.7483704618736676
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        StackingEstimator(estimator=BernoulliNB(alpha=10.0, fit_prior=False)),
    ),
    StackingEstimator(
        estimator=XGBClassifier(
            learning_rate=0.01,
            max_depth=7,
            min_child_weight=16,
            n_estimators=100,
            nthread=1,
            subsample=0.4,
        )
    ),
    MinMaxScaler(),
    ZeroCount(),
    LinearSVC(C=5.0, dual=False, loss="squared_hinge",
              penalty="l1", tol=0.001),
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, "random_state", 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
