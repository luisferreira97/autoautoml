from copy import copy

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import (FunctionTransformer, MaxAbsScaler,
                                   MinMaxScaler, PolynomialFeatures)
from sklearn.svm import LinearSVR
from tpot.builtins import OneHotEncoder, StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE',
                        sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -145.06416257782573
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        OneHotEncoder(minimum_fraction=0.05, sparse=False, threshold=10)
    ),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.9, learning_rate=0.001, loss="ls", max_depth=10,
                      max_features=0.6500000000000001, min_samples_leaf=13, min_samples_split=17, n_estimators=100, subsample=0.15000000000000002)),
    MaxAbsScaler(),
    StackingEstimator(estimator=LinearSVR(C=0.1, dual=True,
                      epsilon=0.0001, loss="epsilon_insensitive", tol=0.0001)),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    MinMaxScaler(),
    AdaBoostRegressor(learning_rate=0.001,
                      loss="exponential", n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
