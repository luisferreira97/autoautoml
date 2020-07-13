import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler, StandardScaler
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -145.63477323108145
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RidgeCV()),
    StandardScaler(),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=1, min_child_weight=9, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.5)),
    SelectPercentile(score_func=f_regression, percentile=13),
    RobustScaler(),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.7000000000000001, min_samples_leaf=20, min_samples_split=11, n_estimators=100)),
    GradientBoostingRegressor(alpha=0.9, learning_rate=0.01, loss="lad", max_depth=2, max_features=0.8500000000000001, min_samples_leaf=12, min_samples_split=16, n_estimators=100, subsample=0.3)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
