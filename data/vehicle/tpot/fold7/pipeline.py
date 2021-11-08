import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE',
                        sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.8576549972302001
exported_pipeline = make_pipeline(
    FastICA(tol=0.9),
    StackingEstimator(estimator=MLPClassifier(
        alpha=0.001, learning_rate_init=0.1)),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=GaussianNB()),
    MinMaxScaler(),
    StackingEstimator(estimator=LinearSVC(C=15.0, dual=False,
                      loss="squared_hinge", penalty="l2", tol=0.001)),
    SelectPercentile(score_func=f_classif, percentile=47),
    KNeighborsClassifier(n_neighbors=79, p=2, weights="distance")
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
