Leaf Classification
==============================
Kaggle "Leaf Classification" database classification project for IFT712.

**Python version >= 3.5 required

For package installation, you may run*: 
> $ pip install -r "requirements.txt"

*In case there is an error regarding "pkg-resources==0.0.0", you may try removing it from the requirements.txt file.
There seem to be a general issue regarding this particular package and virtual environment.

Data Initialisation
------------
Make sure there is a "data" folder as presented in the architecture below, with your raw data into "data/raw".

You may then run: (you need Make to execute Makefile)
> $ make data

Model Training
------------
> $ python main.py [model] [pipeline] [cv_metrics] [evaluate] [data_processing]
> 
>> model : bagging | decision_tree | fconnected | gboost | logit | random_forest | svm
>> 
>> pipeline: simple | cross_validation 
>>  cv_metrics: accuracy | roc_auc_ovr 
>>
>> evaluate: <report> | confusion_matrix 
>>
>> data_processing: <Empty> | simple | fselection | pca_50 | pca_100 | pca_150
> 
> 
> $ python cross_valuation.py [pipeline] [data_processing]
>>
>> pipeline: simple | cross_validation
>>
>> data_processing: <Empty> | simple | fselection | pca_50 | pca_100 | pca_150

Project Organization
------------

    ├── LICENSE
    ├── Makefile            <- Makefile with commands like `make data` or `make train`
    ├── README.md           <- The top-level README for developers using this project.
    ├── data
    │   ├── interim         <- Intermediate data that has been transformed.
    │   ├── processed       <- The final, canonical data sets for modeling.
    │   └── raw             <- The original, immutable data dump.
    │      ├── test.csv     <- testing set from Leaf Classification Competition (Kaggle) 
    │      └── train.csv    <- training set from Leaf Classification Competition (Kaggle)
    │
    ├── models              <- Trained and serialized models, model predictions, or model summaries
    │   ├──  best_estimator <- Best models from GridSearchCV
    │
    ├── notebooks           <- Jupyter notebooks.
    │
    ├── reports             <- Generated analysis as PDF
    │   ├── cross_valuation <- Cross-valuation Result
    │   └── figures         <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt    <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py**          <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                 <- Source code for use in this project.
    │   ├── __init__.py     <- Makes src a Python module
    │   │
    │   ├── data            <- Scripts to generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── models          <- Scripts for the usage of the models
    │   |   ├── bagging_classifier.py
    │   |   ├── decision_tree_classifier.py
    │   |   ├── gradient_boosting_classifier.py
    │   |   ├── logistic_regression_classifier.py
    │   |   ├── ramdom_forest_classifier.py
    │   |   ├── svm_classifier.py
    │   |   └── vanilla_classifier.py
    │   │
    │   └── visualization**  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── cross_valuation.py   <- python3 cross_valuation.py <pipeline> <data_processing>
    └── main.py              <- python3 main.py <model> <pipeline> <cv_metrics> <evaluate> <data_processing>
** Not yet implemented

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
