from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from data.preprocessor import Preprocessor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import logging
import joblib

class ModelTrainer:
    def __init__(self):
        self.model = ImbPipeline([
            ('preprocessing', Preprocessor()),  # Preprocessing step
            ('undersampling', RandomUnderSampler(random_state=42)),  # Use Random UnderSampler instead of SMOTE
            ('classifier', DecisionTreeClassifier())  # Classifier
        ])
        
        # Define parameter grid for hyperparameter tuning
        self.param_grid = {
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__criterion': ['gini', 'entropy']
        }
    
    def train(self, X_train, y_train, X_val, y_val):
        try:
            # Grid search with cross-validation
            grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, 
                                       cv=2, scoring='accuracy', n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            logging.info(f"Best Parameters: {grid_search.best_params_}")

            # Train with best parameters
            self.model = best_model
            self.model.fit(X_train, y_train)
            logging.info(f"Model trained successfully with best parameters")
            
            # Validation predictions and metrics
            val_predictions = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)
            val_report = classification_report(y_val, val_predictions)
            logging.info(f"Validation Accuracy: {val_accuracy:.4f}")
            logging.info("Validation Classification Report:\n" + val_report)
            
            # Save model
            joblib.dump(self.model, 'pipeline_model.pkl')
            logging.info(f"Model saved to pipeline_model.pkl")
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise