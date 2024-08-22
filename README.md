# Data-Science
# Purpose

This repository is aimed at predicting housing prices from the famous Boston housing dataset using an array of machine learning models including Linear Regression (LASSO, Ridge, ELasticNet), Random Forest regression and Extreme Gradient Boosting (XGBoost) regression. 

# Repo Location + Structure

This project repository is located at `https://github.com/petreli12/Data-Science` and is structured as:

        Data-Science/
                    bin/
                    config/
                    notebooks/
                    python_env/
                    scripts/

# Setup

## Installation

A python virtual environment built using:

1. Python 3.8.16
2. A requirements file that contains the minimal requirements to install necessary packages to read/write to S3.

**Step 1: Create virtual environment, activate venv, and install requirements**

``` bash
cd
mkdir -p python_env
cd python_env
virtualenv -p /usr/bin/python3.8 Data-Science/
source ~/python_venv/Data-Science/bin/activate
pip install -r ~/repos/Data-Science/requirements.txt
```

**Step 3: Clone `Data-Science` and install from develop branch**

``` bash
cd
mkdir -p repos
cd ~/repos/
git clone https://github.com/petreli12/Data-Science.git
cd ~/repos/Data-Science/
git checkout develop
pip install .
```


# Run Order

### Binary Executables (`bin/`)

* `run.sh` executes all listed scripts in order.

### Python Scripts (`scripts/`)

* `Peter_Olayemi_Sample_Code.py` - import housing dataset csv, preprocesses the data and carry out exploratory data analysis and then runs baseline and hyperparameter tuned machine learning models. Also returns model evaluation metrics and feature importance.
