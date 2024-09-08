from data.dataloader import DataLoader
from models.train_model import ModelTrainer
from models.predict import ModelPredictor
from sklearn.metrics import accuracy_score, classification_report
import logging

if __name__ == '__main__':
    try:
        # Load and split data
        PATH = r'C:\Users\Juan Zuluaga\Desktop\OrdersModel\src\data\orders.csv'
        data_loader = DataLoader(PATH)
        data = data_loader.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(data)

        # Train the model using pipeline
        trainer = ModelTrainer()
        trainer.train(X_train, y_train, X_val, y_val)
        # Make predictions using pipeline
        predictor = ModelPredictor()
        test_predictions = predictor.predict(X_test)

        # Evaluate predictions
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_report = classification_report(y_test, test_predictions)

        # Log and print metrics
        logging.info(f"Test Accuracy: {test_accuracy:.4f}")
        logging.info("Classification Report:\n" + test_report)

        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("Classification Report:\n" + test_report)
    
    except Exception as e:
        logging.critical(f"Critical error in main pipeline: {e}")