from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class Preprocessor(BaseEstimator, TransformerMixin):

    """
    A custom preprocessor for numerical data that handles missing values,
    scales numerical features, and creates new features.

    Attributes:
        imputer_mean (SimpleImputer): Imputer for filling missing values with the mean.
        encoder (OneHotEncoder): Encoder for categorical features.
        scaler (StandardScaler): Scaler for numerical features.
        feature_engineer (FeatureEngineer): Feature engineering transformer.

    Methods:
        fit(X, y=None): Fits the preprocessor on the given data.
        transform(X): Transforms the data using the fitted preprocessor.
    """
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        """
        Fits the preprocessor on the given data.

        Args:
            X (pd.DataFrame): The input data.
            y (pd.Series or None): The target variable. Not used in this method.

        Returns:
            self: The fitted Preprocessor instance.
        """
        # Drop columns that are not needed
        X = X.drop(columns=['CREATED_AT', 'COUNTRY', 'CITY', 'ORDER_ID'], errors='ignore')
        
        # Identify numerical columns
        self.num_cols = X.select_dtypes(include=['float64', 'int64']).columns
        
        # Handle missing values manually: replace with mean
        self.mean_values = X[self.num_cols].mean()
        self.std_values = X[self.num_cols].std()
        
        return self

    def transform(self, X, y=None):
        """
        Transforms the data using the fitted preprocessor.

        Args:
            X (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The transformed data.
        """
        X = X.drop(columns=['CREATED_AT', 'COUNTRY', 'CITY', 'ORDER_ID'], errors='ignore')
        X_num = X[self.num_cols]
        X_num = X_num.fillna(self.mean_values)
        X_num_scaled = (X_num - self.mean_values) / self.std_values
        
        # Add new feature for distance-to-earnings ratio
        if 'TOTAL_EARNINGS' in X.columns and 'TO_USER_DISTANCE' in X.columns:
            distance_earnings_ratio = X['TO_USER_DISTANCE'] / (X['TOTAL_EARNINGS'] + X['TIP']).replace(0, np.nan)  # Avoid division by zero
            distance_earnings_ratio = distance_earnings_ratio.fillna(0)  # Handle any NaN values
            X_num_scaled = np.hstack((X_num_scaled, distance_earnings_ratio.values.reshape(-1, 1)))
        
        # Create DataFrame with new column names
        feature_names = list(self.num_cols) + ['distance_earnings_ratio']
        X_processed = pd.DataFrame(data=X_num_scaled, columns=feature_names)
        
        return X_processed