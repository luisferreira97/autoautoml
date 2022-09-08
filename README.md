# Benchmark of AutoML tools

## Computational experiments for the paper "A Comparison of AutoML Tools for Machine Learning, Deep Learning and XGBoost" (IJCNN 2021)

### [ResearchGate](https://bit.ly/30gxcfs)
### [DOI](https://bit.ly/3F8mM0t)

To cite this work please use:

```
@inproceedings{DBLP:conf/ijcnn/FerreiraPMPC21,
  author    = {Lu{\'{\i}}s Ferreira and
               Andr{\'{e}} Luiz Pilastri and
               Carlos Manuel Martins and
               Pedro Miguel Pires and
               Paulo Cortez},
  title     = {A Comparison of AutoML Tools for Machine Learning, Deep Learning and
               XGBoost},
  booktitle = {International Joint Conference on Neural Networks, {IJCNN} 2021, Shenzhen,
               China, July 18-22, 2021},
  pages     = {1--8},
  publisher = {{IEEE}},
  year      = {2021},
  url       = {https://doi.org/10.1109/IJCNN52387.2021.9534091},
  doi       = {10.1109/IJCNN52387.2021.9534091},
}

```


## Folder Description

1. The code that was used to generate all the benchmark models is inside the **data** folder and its subfolders.
2. Inside the **data** folder, there is a subfolder for each of the datasets used for the benchmark.
3. Inside the **datasets** subfolders, there is one subfolder for each AutoML tool used for that dataset.
4. Inside the **tools** subfolders, there is the script used to generate the ML models and the resulting metadata (e.g., model leaderboards, performance metrics)

## Folder Structure

```
project
└───aux_functions: scripts to divide the original datasets into folds
    │   join_data.py
    │   split_data.py
└───docs: PDF of the IJCNN paper and other documentation (e.g., list of OpenML datasets, AutoML tools descriptions)
└───data:
    └───dataset A
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
    └───dataset B
    └───dataset C
    └───.....
│   README.md
│   requirements.txt
```
