from data.preprocessor import Preprocessor
import numpy as np
import pandas as pd
from data.dataloader import DataLoader
from models.train_model import ModelTrainer




def test_preprocessor_initialization():
    """
    Tests if the Preprocessor is properly initialized with its components.
    """
    preprocessor = Preprocessor()
    assert preprocessor.scaler is not None, "Scaler should be initialized"

def test_preprocessor_fit(X_train):
    """
    Tests if the Preprocessor's fit method sets necessary attributes.
    """
    preprocessor = Preprocessor()
    preprocessor.fit(X_train)
    assert hasattr(preprocessor, 'mean_values'), "mean_values should be set after fit"
    assert hasattr(preprocessor, 'std_values'), "std_values should be set after fit"

def test_preprocessor_transform(X_train):
    """
    Tests if the transform method correctly transforms the data.
    """
    preprocessor = Preprocessor()
    preprocessor.fit(X_train)
    X_transformed = preprocessor.transform(X_train)
    assert X_transformed.shape[1] == len(preprocessor.num_cols) + 1, "Number of columns should match expected"
    assert 'distance_earnings_ratio' in X_transformed.columns, "distance_earnings_ratio should be in the transformed data"

def test_missing_values_handling():
    """
    Tests if missing values are handled correctly in the Preprocessor.
    """
    # Create a sample DataFrame with missing values
    X_sample = pd.DataFrame({
        'TO_USER_DISTANCE': [5.0, 2.0, np.nan],
        'TOTAL_EARNINGS': [1000.0, 1500.0, np.nan],
        'TIP': [10.0, 15.0, np.nan]
    })
    preprocessor = Preprocessor()
    preprocessor.fit(X_sample)
    X_transformed = preprocessor.transform(X_sample)
    assert X_transformed.notna().all().all(), "There should be no missing values after transformation"

def test_model_training(X_train, y_train,X_val,y_val):
    """
    Tests if the model training function trains the model and sets attributes correctly.
    """
    trainer = ModelTrainer()
    trainer.train(X_train, y_train, X_val, y_val)
    assert trainer.model is not None, "Model should be trained and not None"
    assert hasattr(trainer.model, 'best_estimator_'), "Trained model should have best_estimator_ attribute"

def test_model_prediction(X_train,y_train,X_val,y_val):
    """
    Tests if the model makes predictions with the correct shape and values.
    """
    trainer = ModelTrainer()
    trainer.train(X_train, y_train, X_val, y_val)
    predictions = trainer.model.predict(X_val)
    assert len(predictions) == len(y_val), "Number of predictions should match the number of validation labels"
    assert set(predictions) <= set(y_val.unique()), "Predictions should be within the range of target values"
