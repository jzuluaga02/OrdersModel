import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report
from data.preprocessor import Preprocessor  # Adjust the import based on your module
from data.dataloader import DataLoader
from models.train_model import ModelTrainer


def generate_sample_data():
    """
    Generates sample data for testing purposes.
    
    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Sample features dataset.
            - pd.Series: Sample target variable.
    """
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')
    return X_df, y_series


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


def main():
    # Generate sample data
    X, y = generate_sample_data()
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize the Preprocessor and ModelTrainer
    preprocessor = Preprocessor()
    trainer = ModelTrainer()
    
    # Run unit tests
    print("Running Preprocessor Initialization Test...")
    test_preprocessor_initialization()
    
    print("Running Preprocessor Fit Test...")
    test_preprocessor_fit()
    
    print("Running Preprocessor Transform Test...")
    test_preprocessor_transform()
    
    print("Running Missing Values Handling Test...")
    test_missing_values_handling()
    
    print("Running Data Load Test...")
    # Assuming 'data.csv' is available or use the generated data
    # X_loaded, y_loaded = load_data('data.csv')
    # test_load_data()
    
    print("Running Model Training Test...")
    test_model_training()
    
    print("Running Model Prediction Test...")
    test_model_prediction()

if __name__ == "__main__":
    main()

