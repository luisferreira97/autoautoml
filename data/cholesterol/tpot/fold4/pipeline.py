import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv("PATH/TO/DATA/FILE",
                        sep="COLUMN_SEPARATOR", dtype=np.float64)
features = tpot_data.drop("target", axis=1)
training_features, testing_features, training_target, testing_target = train_test_split(
    features, tpot_data["target"], random_state=42
)

imputer = SimpleImputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Average CV score on the training set was: -36.07369004275705
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.8, tol=0.001)),
    MinMaxScaler(),
    StackingEstimator(estimator=RidgeCV()),
    StackingEstimator(estimator=RidgeCV()),
    StackingEstimator(
        estimator=DecisionTreeRegressor(
            max_depth=3, min_samples_leaf=3, min_samples_split=18
        )
    ),
    ExtraTreesRegressor(
        bootstrap=True,
        max_features=0.4,
        min_samples_leaf=6,
        min_samples_split=9,
        n_estimators=100,
    ),
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, "random_state", 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
