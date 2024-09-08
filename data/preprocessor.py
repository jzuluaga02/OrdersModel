from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer_mean = SimpleImputer(strategy='mean')
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Dropping 'CREATED_AT' before fitting
        X = X.drop(columns=['CREATED_AT'])

        # Separating numerical and categorical columns for fitting
        num_cols = X.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = X.select_dtypes(include=['object']).columns

        self.imputer_mean.fit(X[num_cols])
        self.encoder.fit(X[cat_cols])

        return self

    def transform(self, X, y=None):
        # Dropping 'CREATED_AT' before transforming
        X = X.drop(columns=['CREATED_AT'])

        # Separating numerical and categorical columns
        num_cols = X.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = X.select_dtypes(include=['object']).columns

        # Applying imputations and scaling
        X_num = self.imputer_mean.transform(X[num_cols])
        X_cat = self.encoder.transform(X[cat_cols]).toarray()  # Convert sparse matrix to dense

        # Scaling numerical data
        X_num_scaled = self.scaler.fit_transform(X_num)

        # Concatenating processed numerical and categorical data
        X_processed = pd.DataFrame(data=np.hstack((X_num_scaled, X_cat)))

        return X_processed
