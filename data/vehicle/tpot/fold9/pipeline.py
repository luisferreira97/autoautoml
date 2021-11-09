import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv("PATH/TO/DATA/FILE",
                        sep="COLUMN_SEPARATOR", dtype=np.float64)
features = tpot_data.drop("target", axis=1)
training_features, testing_features, training_target, testing_target = train_test_split(
    features, tpot_data["target"], random_state=42
)

# Average CV score on the training set was: 0.8684448939690779
exported_pipeline = make_pipeline(
    FastICA(tol=0.4),
    RobustScaler(),
    MLPClassifier(alpha=0.001, learning_rate_init=0.01),
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, "random_state", 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
