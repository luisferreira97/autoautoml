import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv("PATH/TO/DATA/FILE",
                        sep="COLUMN_SEPARATOR", dtype=np.float64)
features = tpot_data.drop("target", axis=1)
training_features, testing_features, training_target, testing_target = train_test_split(
    features, tpot_data["target"], random_state=42
)

# Average CV score on the training set was: -2.232903679252158
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ElasticNetCV(
        l1_ratio=0.35000000000000003, tol=0.001)),
    GradientBoostingRegressor(
        alpha=0.75,
        learning_rate=0.1,
        loss="huber",
        max_depth=4,
        max_features=0.2,
        min_samples_leaf=16,
        min_samples_split=15,
        n_estimators=100,
        subsample=0.8,
    ),
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, "random_state", 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
