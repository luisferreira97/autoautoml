import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.linear_model import (ElasticNetCV, LassoLarsCV, RidgeCV,
                                  SGDRegressor)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from tpot.builtins import OneHotEncoder, StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE',
                        sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -0.2199229013614184
exported_pipeline = make_pipeline(
    SelectFwe(score_func=f_regression, alpha=0.005),
    StackingEstimator(estimator=ElasticNetCV(
        l1_ratio=0.30000000000000004, tol=0.1)),
    StackingEstimator(estimator=SGDRegressor(alpha=0.001, eta0=0.1, fit_intercept=False,
                      l1_ratio=0.75, learning_rate="constant", loss="huber", penalty="elasticnet", power_t=0.5)),
    StandardScaler(),
    StackingEstimator(estimator=RidgeCV()),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=RidgeCV()),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.95, learning_rate=0.01, loss="quantile",
                      max_depth=5, max_features=0.05, min_samples_leaf=5, min_samples_split=15, n_estimators=100, subsample=1.0)),
    OneHotEncoder(minimum_fraction=0.25, sparse=False, threshold=10),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    ElasticNetCV(l1_ratio=0.2, tol=0.001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
