This repository is Avito-Kaggle compertition in 2018:

Target of the task is predict of porbability of the demand (regression task)

1. [Intro](intro.ipynb) - downloading and unpacking info from Kaggle
2. [Prepocessing](preprocexxing.ipynb) clean and preprocess feature on the base of [Target Encoding](utils.py) on common 
3. [Description](description.ipynb) - overview of the main features in dataset
4. [Catboost model](cat_boost.ipynb) - cat boost model - not cool result
5. [Checking tree strategy](Tree Strategy.ipynb):
    * common way
    * use two model - one -classification for zero porbability and second for other part of dataset
    * use only classification and use porbaility as weights for the model
6. [lightgbm model](Tfidf-Ridge-Descript.ipynb) - lightgbm model on the all [features](preprocexxing.ipynb)

Result solution - blending of own one and other solutions from kaggle - 561 / 1917