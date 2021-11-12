# Benchmark of AutoML tools

## Computational experiments for the paper "A Comparison of AutoML Tools for Machine Learning, Deep Learning and XGBoost" (IJCNN 2021)

### [Research Gate](https://www.researchgate.net/publication/354721903_A_Comparison_of_AutoML_Tools_for_Machine_Learning_Deep_Learning_and_XGBoost)
### [DOI](http://dx.doi.org/10.1109/IJCNN52387.2021.9534091)


## Folder Description

### The code that was used to generate all the benchmark models is inside the **data** folder and its subfolders.
### Inside the **data** folder, there is a subfolder for each of the datasets used for the benchmark.
### Inside the **datasets** subfolders, there is one subfolder for each AutoML tool used for that dataset.
### Inside the **tools** subfolders, there is the script used to generate the ML models and the resulting metadata (e.g., model leaderboards, performance metrics)

## Folder Structure

```
project
└───aux_functions: scripts to divide the original datasets into folds
    │   join_data.py
    │   split_data.py
└───docs: PDF of the IJCNN paper and other documentation (e.g., list of OpenML datasets, AutoML tools descriptions)
└───data:
    │   dataset A
    └───AutoML Tool A
        │   run.py: script to run the experiment
        └─── fold 1
             |   model leaderboard
             |   performance metrics
             |   other metadata files
        └─── fold 2
        └─── fold 3
        └─── ....
    └───AutoML Tool B
    └───AutoML Tool C
    └───......
    │   dataset B
    │   dataset C
    │   .....
│   README.md
│   requirements.txt
```
