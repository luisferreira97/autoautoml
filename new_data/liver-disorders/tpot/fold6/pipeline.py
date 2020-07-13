import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -2.254397893423648
exported_pipeline = make_pipeline(
    make_union(
        StandardScaler(),
        FunctionTransformer(copy)
    ),
    StackingEstimator(estimator=LinearSVR(C=0.1, dual=False, epsilon=0.001, loss="squared_epsilon_insensitive", tol=0.1)),
    GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss="lad", max_depth=4, max_features=0.7500000000000001, min_samples_leaf=14, min_samples_split=12, n_estimators=100, subsample=0.5)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
