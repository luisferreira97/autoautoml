import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import ZeroCount
from tpot.export_utils import set_param_recursive
from xgboost import XGBClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv("PATH/TO/DATA/FILE",
                        sep="COLUMN_SEPARATOR", dtype=np.float64)
features = tpot_data.drop("target", axis=1)
training_features, testing_features, training_target, testing_target = train_test_split(
    features, tpot_data["target"], random_state=42
)

# Average CV score on the training set was: 0.8401274936941068
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    RFE(
        estimator=ExtraTreesClassifier(
            criterion="gini", max_features=0.1, n_estimators=100
        ),
        step=0.3,
    ),
    RFE(
        estimator=ExtraTreesClassifier(
            criterion="gini", max_features=0.35000000000000003, n_estimators=100
        ),
        step=0.2,
    ),
    SelectFromModel(
        estimator=ExtraTreesClassifier(
            criterion="gini", max_features=0.15000000000000002, n_estimators=100
        ),
        threshold=0.0,
    ),
    ZeroCount(),
    XGBClassifier(
        learning_rate=0.01,
        max_depth=4,
        min_child_weight=1,
        n_estimators=100,
        nthread=1,
        subsample=0.6500000000000001,
    ),
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, "random_state", 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
