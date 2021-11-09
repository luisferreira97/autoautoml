import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer
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

# Average CV score on the training set was: -145.38165000203776
exported_pipeline = make_pipeline(
    make_union(
        Binarizer(threshold=0.5),
        StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.1, tol=0.0001)),
    ),
    Binarizer(threshold=0.05),
    StackingEstimator(
        estimator=RandomForestRegressor(
            bootstrap=True,
            max_features=0.25,
            min_samples_leaf=19,
            min_samples_split=8,
            n_estimators=100,
        )
    ),
    PCA(iterated_power=10, svd_solver="randomized"),
    StackingEstimator(
        estimator=DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=4, min_samples_split=20
        )
    ),
    GradientBoostingRegressor(
        alpha=0.75,
        learning_rate=0.01,
        loss="lad",
        max_depth=5,
        max_features=0.7500000000000001,
        min_samples_leaf=3,
        min_samples_split=19,
        n_estimators=100,
        subsample=0.7500000000000001,
    ),
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, "random_state", 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
