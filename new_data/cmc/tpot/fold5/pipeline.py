import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.554367102717963
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.001, max_depth=4, max_features=0.9500000000000001, min_samples_leaf=16, min_samples_split=12, n_estimators=100, subsample=0.25)),
    RobustScaler(),
    StackingEstimator(estimator=BernoulliNB(alpha=0.001, fit_prior=False)),
    RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.35000000000000003, n_estimators=100), step=0.7000000000000001),
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.001, max_depth=8, max_features=0.5, min_samples_leaf=2, min_samples_split=9, n_estimators=100, subsample=0.6000000000000001)),
    MLPClassifier(alpha=0.0001, learning_rate_init=0.001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
