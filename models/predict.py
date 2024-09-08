import joblib
import os
import logging
from data.dataloader import FileLoadingError


class ModelPredictor:
    def __init__(self, model_path='pipeline_model.pkl'):
        try:
            if not os.path.exists(model_path):
                raise FileLoadingError(f"Model file not found: {model_path}")
            self.model = joblib.load(model_path)
            logging.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    def predict(self, X_test):
        try:
            predictions = self.model.predict(X_test)
            logging.info(f"Predictions made successfully")
            return predictions
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise
