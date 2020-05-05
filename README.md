# AutoAutoML Package

## Package for running and testing several AutoML tools

### How to run examples (using the *churn* dataset)

#### Auto-Gluon
```
from autoautoml import autogluon
algo = autogluon.AUTOGLUON()
ex = algo.run_example()
```

#### Auto-Keras
```
from autoautoml import autokeras
algo = autokeras.AUTOKERAS()
ex = algo.run_example()
```

#### Auto-Sklearn
```
from autoautoml import autosklearn
algo = autosklearn.AUTOSKLEARN()
ex = algo.run_example()
```

#### Auto-Weka
**Currently unavailable (pyautoweka doesn't work)**

#### H2O AutoML
```
from autoautoml import h2o_automl
algo = h2o_automl.H2O()
ex = algo.run_example()
```

#### TPOT
```
from autoautoml import Tpot
algo = Tpot.TPOT()
ex = algo.run_example()
```

#### Rminer
```
source ./autoautoml/rminer_automl.R
```

### TransmogrifAI
**Coming soon (Scala language)**
