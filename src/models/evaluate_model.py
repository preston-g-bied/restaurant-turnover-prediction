# src/models/evaluate_model.py

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
import joblib
import pickle
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', 'model_evaluation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('model_evaluation')

class ModelEvaluator:
    """
    A class to evaluate trained models and visualize their performance.
    """
    
    def __init__(self):
        """
        Initialize the model evaluator.
        
        Parameters:
        -----------
        models_dir : str
            Path to the directory containing trained models
        reports_dir : str
            Path to save evaluation reports
        data_dir : str
            Path to the directory containing processed data
        """
        self.models_dir = os.path.join(project_root, 'models')
        self.reports_dir = os.path.join(project_root, 'reports')
        self.data_dir = os.path.join(project_root, 'data', 'processed')
        
        # create reports directory if it doesn't exist
        os.makedirs(os.path.join(self.reports_dir, 'figures'), exist_ok=True)
        
        # load model info
        self.model_info = None
        self.load_model_info()
        
        logger.info("ModelEvaluator initialized")
    
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
    
    def load_data(self):
        """
        Load processed training data.
        
        Returns:
        --------
        X_train, y_train
        """
        try:
            train_path = os.path.join(self.data_dir, 'train_processed.csv')
            train_data = pd.read_csv(train_path)
            
            # separate features and target
            y_train = train_data['Annual Turnover']
            X_train = train_data.drop('Annual Turnover', axis=1)
            
            logger.info(f"Loaded training data with shape: {X_train.shape}")
            return X_train, y_train
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None, None
    
    def plot_residuals(self, model, X, y):
        """
        Plot residuals to check for patterns.
        
        Parameters:
        -----------
        model : object
            Trained model
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series
            True values
        """
        # Generate predictions
        y_pred = model.predict(X)
        
        # Calculate residuals
        residuals = y - y_pred
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Residuals vs. predicted values
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='-')
        axes[0].set_title('Residuals vs Predicted Values')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        
        # Histogram of residuals
        axes[1].hist(residuals, bins=30, alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='-')
        axes[1].set_title('Distribution of Residuals')
        axes[1].set_xlabel('Residual Value')
        axes[1].set_ylabel('Frequency')
        
        # Add stats to the histogram
        mean_resid = residuals.mean()
        std_resid = residuals.std()
        axes[1].text(0.05, 0.95, f'Mean: {mean_resid:.2f}\nStd Dev: {std_resid:.2f}',
                   transform=axes[1].transAxes, verticalalignment='top')
        
        plt.suptitle('Residual Analysis', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.reports_dir, 'figures', 'residuals_analysis.png'))
        logger.info("Saved residuals analysis plot")
        
        plt.close()
    
    def plot_actual_vs_predicted(self, model, X, y):
        """
        Plot actual values against predicted values.
        
        Parameters:
        -----------
        model : object
            Trained model
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series
            True values
        """
        # Generate predictions
        y_pred = model.predict(X)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot points
        plt.scatter(y, y_pred, alpha=0.5)
        
        # Plot perfect prediction line
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        # Add metrics to plot
        plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}\nR²: {r2:.4f}',
                transform=plt.gca().transAxes, verticalalignment='top')
        
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.reports_dir, 'figures', 'actual_vs_predicted.png'))
        logger.info("Saved actual vs predicted plot")
        
        plt.close()
    
    def plot_feature_importance(self, model, X):
        """
        Plot feature importances if available.
        
        Parameters:
        -----------
        model : object
            Trained model
        X : pandas.DataFrame
            Feature matrix
        """
        # Check if model has feature importances
        if not hasattr(model, 'feature_importances_'):
            logger.info("Model does not have feature importances attribute, skipping plot")
            return
        
        # Get feature importances
        importances = pd.Series(model.feature_importances_, index=X.columns)
        
        # Sort importances
        importances = importances.sort_values(ascending=False)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Plot horizontal bar chart
        importances.plot(kind='barh')
        
        plt.title('Feature Importances')
        plt.xlabel('Importance')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.reports_dir, 'figures', 'feature_importance.png'))
        logger.info("Saved feature importance plot")
        
        # Also save as CSV
        importances.to_csv(os.path.join(self.reports_dir, 'feature_importance.csv'))
        logger.info("Saved feature importance CSV")
        
        plt.close()
    
    def plot_learning_curve(self, model, X, y):
        """
        Plot learning curve to diagnose overfitting/underfitting.
        
        Parameters:
        -----------
        model : object
            Trained model
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series
            True values
        """
        # Define train sizes
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Calculate learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=5,
            scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        
        # Calculate mean and std for train and test scores
        train_mean = -np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = -np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot training and validation curves
        plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training RMSE')
        plt.plot(train_sizes, test_mean, 'o-', color='g', label='Validation RMSE')
        
        # Add shaded regions for standard deviation
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
        
        plt.title('Learning Curve')
        plt.xlabel('Training Set Size')
        plt.ylabel('RMSE')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.reports_dir, 'figures', 'learning_curve.png'))
        logger.info("Saved learning curve plot")
        
        plt.close()
    
    def generate_error_analysis(self, model, X, y):
        """
        Generate error analysis to understand where the model performs poorly.
        
        Parameters:
        -----------
        model : object
            Trained model
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series
            True values
        """
        # Generate predictions
        y_pred = model.predict(X)
        
        # Calculate absolute errors
        abs_errors = np.abs(y - y_pred)
        
        # Create a dataframe with true values, predictions, and errors
        error_df = pd.DataFrame({
            'Actual': y,
            'Predicted': y_pred,
            'AbsoluteError': abs_errors,
            'RelativeError': abs_errors / y * 100
        })
        
        # Add features
        for col in X.columns:
            error_df[col] = X[col]
        
        # Sort by absolute error
        error_df = error_df.sort_values('AbsoluteError', ascending=False)
        
        # Save the top 100 errors
        error_df.head(100).to_csv(os.path.join(self.reports_dir, 'top_100_errors.csv'))
        logger.info("Saved top 100 errors analysis")
        
        # Analyze errors by feature ranges/bins
        error_analysis = {}
        
        # For numeric features, analyze errors in different ranges
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        
        for feature in numeric_features:
            # Create bins
            bins = pd.qcut(X[feature], q=5, duplicates='drop')
            
            # Group by bins and compute mean error
            error_by_bin = error_df.groupby(bins)['AbsoluteError'].mean()
            
            error_analysis[feature] = error_by_bin
        
        # Create a summary plot for error analysis by feature
        plt.figure(figsize=(15, 10))
        
        # Plot top 5 features with highest error variance
        feature_error_std = [
            (feature, error_df.groupby(pd.qcut(X[feature], q=5, duplicates='drop'))['AbsoluteError'].std().mean())
            for feature in numeric_features
        ]
        
        top_features = sorted(feature_error_std, key=lambda x: x[1], reverse=True)[:5]
        
        for i, (feature, _) in enumerate(top_features):
            plt.subplot(2, 3, i+1)
            
            # Create bins
            try:
                bins = pd.qcut(X[feature], q=5, duplicates='drop')
                
                # Group by bins and compute mean error
                error_by_bin = error_df.groupby(bins)['AbsoluteError'].mean()
                
                # Plot
                plt.bar(range(len(error_by_bin)), error_by_bin.values)
                plt.title(f'Error by {feature}')
                plt.xticks(range(len(error_by_bin)), [str(x) for x in error_by_bin.index.categories], rotation=45)
                plt.xlabel(feature)
                plt.ylabel('Mean Absolute Error')
            except Exception as e:
                logger.warning(f"Error creating plot for {feature}: {str(e)}")
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.reports_dir, 'figures', 'error_analysis_by_feature.png'))
        logger.info("Saved error analysis by feature plot")
        
        plt.close()
        
        return error_df, error_analysis
    
    def generate_evaluation_report(self):
        """
        Generate a comprehensive evaluation report.
        """
        logger.info("Generating evaluation report")
        
        # Load model and data
        model = self.load_model()
        X_train, y_train = self.load_data()
        
        if model is None or X_train is None:
            logger.error("Failed to load model or data, cannot generate report")
            return
        
        # Generate predictions
        y_pred = model.predict(X_train)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        mae = mean_absolute_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        
        # Create summary report
        report = {
            'Model Name': self.model_info['model_name'],
            'Model Parameters': self.model_info['model_params'],
            'RMSE': rmse,
            'MAE': mae,
            'R-squared': r2,
            'Feature Count': len(X_train.columns),
            'Training Samples': len(X_train)
        }
        
        # Save report
        with open(os.path.join(self.reports_dir, 'evaluation_summary.txt'), 'w') as f:
            f.write("Model Evaluation Summary\n")
            f.write("======================\n\n")
            
            for key, value in report.items():
                if key == 'Model Parameters':
                    f.write(f"{key}:\n")
                    for param, val in value.items():
                        f.write(f"  {param}: {val}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        logger.info("Saved evaluation summary")
        
        # Generate visualizations
        self.plot_residuals(model, X_train, y_train)
        self.plot_actual_vs_predicted(model, X_train, y_train)
        self.plot_feature_importance(model, X_train)
        self.plot_learning_curve(model, X_train, y_train)
        
        # Perform error analysis
        error_df, error_analysis = self.generate_error_analysis(model, X_train, y_train)
        
        logger.info("Evaluation report generation complete")
        
        return report, error_df, error_analysis

if __name__ == "__main__":
    # Run the model evaluation
    evaluator = ModelEvaluator()
    report, error_df, error_analysis = evaluator.generate_evaluation_report()