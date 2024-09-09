### Decision Tree Model for Predicting Order Acceptance

## Project Overview
This project aims to build and optimize a machine learning model to predict whether a delivery order will be accepted or not (TAKEN). The model uses various features such as distance, earnings, and time-related data. The implementation includes a preprocessing pipeline, model training with hyperparameter tuning, and evaluation metrics.

## Features and Strategy
# Features Used:

Numeric features like distance and earnings
Time-based features like day_of_week and hour_of_day
Derived features such as distance_earnings_ratio
Model: Random Forest Classifier (initially Decision Tree)

# Preprocessing:

Custom scaling and imputation
Feature engineering with time-based features
Imbalanced Data Handling: SMOTE (Synthetic Minority Over-sampling Technique)


## Usage

1. git clone https://github.com/jzuluaga02/OrdersModel
cd OrdersModel

2. pip install -r requirements.txt

3. python main.py


## Files
- main.py: Entry point for training and evaluating the model.
- preprocessor.py: Data preprocessing, including feature engineering.
- model_trainer.py: Contains the Random Forest model pipeline.
- load_model.py: loads the model logs any errors.
- predict_model.py: Loads the model and predicts the giving set in the main
- main_test.py: Unit tests for preprocessing and training.

## Future Improvements
- Experiment with other algorithms (e.g., Gradient Boosting).
- Improve feature selection based on feature importance.
- Optimize performance with larger datasets.
