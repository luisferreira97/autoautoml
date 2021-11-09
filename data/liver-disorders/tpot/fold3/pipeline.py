import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from xgboost import XGBRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv("PATH/TO/DATA/FILE",
                        sep="COLUMN_SEPARATOR", dtype=np.float64)
features = tpot_data.drop("target", axis=1)
training_features, testing_features, training_target, testing_target = train_test_split(
    features, tpot_data["target"], random_state=42
)

# Average CV score on the training set was: -2.2393495263696876
exported_pipeline = make_pipeline(
    StackingEstimator(
        estimator=XGBRegressor(
            learning_rate=0.001,
            max_depth=3,
            min_child_weight=11,
            n_estimators=100,
            nthread=1,
            objective="reg:squarederror",
            subsample=0.45,
        )
    ),
    MinMaxScaler(),
    RandomForestRegressor(
        bootstrap=False,
        max_features=0.4,
        min_samples_leaf=3,
        min_samples_split=3,
        n_estimators=100,
    ),
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, "random_state", 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
