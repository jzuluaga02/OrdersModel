import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
import logging


class FileLoadingError(Exception):
    pass

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            if not os.path.exists(self.file_path):
                raise FileLoadingError(f"File not found: {self.file_path}")
            data = pd.read_csv(self.file_path,encoding='ISO-8859-1')
            logging.info(f"Data loaded successfully from {self.file_path}")
            return data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def split_data(self, data, target_variable='TAKEN', test_size=0.2, val_size=0.1):
        try:
            X = data.drop(target_variable, axis=1)
            y = data[target_variable]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)
            logging.info(f"Data split into training, validation, and testing sets successfully")
            return X_train, X_val, X_test, y_train, y_val, y_test
        except Exception as e:
            logging.error(f"Error splitting data: {e}")
            raise