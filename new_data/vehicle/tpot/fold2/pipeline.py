import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.8636913680548279
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StandardScaler(),
    StackingEstimator(estimator=LinearSVC(C=15.0, dual=False, loss="squared_hinge", penalty="l2", tol=0.0001)),
    StackingEstimator(estimator=LinearSVC(C=15.0, dual=False, loss="squared_hinge", penalty="l1", tol=0.0001)),
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.1, max_depth=6, max_features=0.15000000000000002, min_samples_leaf=19, min_samples_split=4, n_estimators=100, subsample=0.5)),
    DecisionTreeClassifier(criterion="gini", max_depth=9, min_samples_leaf=2, min_samples_split=8)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
