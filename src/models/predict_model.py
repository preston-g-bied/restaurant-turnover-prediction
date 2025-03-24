#!/usr/bin/env python
# src/models/predict_model.py

import os
import logging
import numpy as np
import pandas as pd
import joblib
import pickle
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', 'prediction.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('prediction')

class ModelPredictor:
    """
    A class to generate predictions using a trained model.
    """
    
    def __init__(self):
        """
        Initialize the model predictor.
        
        Parameters:
        -----------
        models_dir : str
            Path to the directory containing trained models
        data_dir : str
            Path to the directory containing processed data
        submission_dir : str
            Path to save submission files
        """
        self.models_dir = os.path.join(project_root, 'models')
        self.data_dir = os.path.join(project_root, 'data', 'processed')
        self.submission_dir = os.path.join(project_root, 'data', 'submissions')
        
        # create submission directory if it doesn't exist
        os.makedirs(self.submission_dir, exist_ok=True)
        
        # Load model info
        self.model_info = None
        self.load_model_info()
        
        logger.info("ModelPredictor initialized")
    
    def load_model_info(self):
        """
        Load model information from the saved file.
        """
        try:
            model_info_path = os.path.join(self.models_dir, 'model_info.pkl')
            with open(model_info_path, 'rb') as f:
                self.model_info = pickle.load(f)
            
            logger.info(f"Loaded model info: {self.model_info['model_name']}")
        except Exception as e:
            logger.error(f"Error loading model info: {str(e)}")
            self.model_info = None
    
    def load_model(self):
        """
        Load the best trained model.
        
        Returns:
        --------
        object
            Trained model
        """
        if self.model_info is None:
            logger.error("Model info not loaded, cannot determine which model to load")
            return None
        
        try:
            model_path = os.path.join(self.models_dir, f"{self.model_info['model_name']}_final.pkl")
            model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def load_test_data(self):
        """
        Load processed test data.
        
        Returns:
        --------
        pandas.DataFrame, pandas.Series
            Test features and IDs
        """
        try:
            test_path = os.path.join(self.data_dir, 'test_processed.csv')
            test_data = pd.read_csv(test_path)
            
            # Extract IDs and features
            test_ids = test_data['Registration Number']
            X_test = test_data.drop('Registration Number', axis=1)
            
            logger.info(f"Loaded test data with shape: {X_test.shape}")
            return X_test, test_ids
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            return None, None
    
    def generate_predictions(self, model, X_test):
        """
        Generate predictions using the trained model.
        
        Parameters:
        -----------
        model : object
            Trained model
        X_test : pandas.DataFrame
            Test features
            
        Returns:
        --------
        numpy.ndarray
            Predicted values
        """
        try:
            # check if all required features are present
            if self.model_info and 'features' in self.model_info:
                required_features = self.model_info['features']
                
                # check for missing features
                missing_features = [f for f in required_features if f not in X_test.columns]
                if missing_features:
                    logger.warning(f"Missing features in test data: {missing_features}")
                    # use only available features
                    X_test = X_test[[f for f in required_features if f in X_test.columns]]
            
            # generate predictions
            y_pred = model.predict(X_test)
            
            logger.info(f"Generated predictions with shape: {y_pred.shape}")
            logger.info(f"Prediction summary - Min: {np.min(y_pred):.2f}, Max: {np.max(y_pred):.2f}, Mean: {np.mean(y_pred):.2f}")
            
            return y_pred
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return None
    
    def create_submission(self, test_ids, predictions, filename=None):
        """
        Create submission file.
        
        Parameters:
        -----------
        test_ids : pandas.Series
            Test registration numbers
        predictions : numpy.ndarray
            Predicted values
        filename : str, optional
            Custom filename for the submission
            
        Returns:
        --------
        pandas.DataFrame
            Submission dataframe
        """
        try:
            # Create submission dataframe
            submission = pd.DataFrame({
                'Registration Number': test_ids,
                'Annual Turnover': predictions
            })
            
            # Generate filename if not provided
            if filename is None:
                model_name = self.model_info['model_name'] if self.model_info else 'unknown'
                filename = f"submission_{model_name}.csv"
            
            # Save submission
            submission_path = os.path.join(self.submission_dir, filename)
            submission.to_csv(submission_path, index=False)
            
            logger.info(f"Submission saved to {submission_path}")
            
            return submission
        except Exception as e:
            logger.error(f"Error creating submission: {str(e)}")
            return None
    
    def run(self, filename=None):
        """
        Run the prediction pipeline.
        
        Parameters:
        -----------
        filename : str, optional
            Custom filename for the submission
            
        Returns:
        --------
        pandas.DataFrame
            Submission dataframe
        """
        logger.info("Running prediction pipeline")
        
        # Load model
        model = self.load_model()
        if model is None:
            return None
        
        # Load test data
        X_test, test_ids = self.load_test_data()
        if X_test is None or test_ids is None:
            return None
        
        # Generate predictions
        predictions = self.generate_predictions(model, X_test)
        if predictions is None:
            return None
        
        # Create and save submission
        submission = self.create_submission(test_ids, predictions, filename)
        
        logger.info("Prediction pipeline completed successfully")
        return submission

if __name__ == "__main__":
    # Run the prediction pipeline
    predictor = ModelPredictor()
    submission = predictor.run()