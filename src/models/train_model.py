# src/models/train_model.py

import os
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import pickle
from pathlib import Path
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs/model_training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('model_training')

class ModelTrainer:
    """
    A class to handle all model training steps including model selection, 
    hyperparameter tuning, evaluation, and submission generation.
    """
    def __init__(self, random_state=42):
        """
        Initialize the model trainer.
        
        Parameters:
        -----------
        data_dir : str
            Path to the directory containing processed data
        models_dir : str
            Path to save trained models
        submission_dir : str
            Path to save submission files
        random_state : int
            Random seed for reproducibility
        """
        self.data_dir = os.path.join(project_root, 'data', 'processed')
        self.models_dir = os.path.join(project_root, 'models')
        self.submission_dir = os.path.join(project_root, 'data', 'submissions')
        self.random_state = random_state
        
        # create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)
        os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)

        # model dictionary to store trained models
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = float('inf')

        logger.info('ModelTrainer initialized')

    def load_data(self):
        """
        Load processed training and test data.
        
        Returns:
        --------
        X_train, y_train, X_test, test_ids
        """
        logger.info('Loading processed data...')

        train_path = os.path.join(self.data_dir, 'train_processed.csv')
        test_path = os.path.join(self.data_dir, 'test_processed.csv')

        # load training data
        train_data = pd.read_csv(train_path)
        logger.info(f'Loaded training data with shape: {train_data.shape}')

        # separate features and target
        y_train = train_data['Annual Turnover']
        X_train = train_data.drop('Annual Turnover', axis=1)

        # load test data
        test_data = pd.read_csv(test_path)
        logger.info(f'Loaded test data with shape: {test_data.shape}')

        # extract test IDs and features
        test_ids = test_data['Registration Number']
        X_test = test_data.drop('Registration Number', axis=1)

        # validate that training and test data have the same features
        train_features = set(X_train.columns)
        test_features = set(X_test.columns)
        
        if train_features != test_features:
            missing_in_test = train_features - test_features
            missing_in_train = test_features - train_features
            
            if missing_in_test:
                logger.warning(f"Features in train but not in test: {missing_in_test}")
            
            if missing_in_train:
                logger.warning(f"Features in test but not in train: {missing_in_train}")
            
            # get common features
            common_features = list(train_features.intersection(test_features))
            logger.info(f"Using {len(common_features)} common features")
            
            X_train = X_train[common_features]
            X_test = X_test[common_features]

        logger.info('Data loading complete')
        return X_train, y_train, X_test, test_ids
    
    def initialize_models(self):
        """
        Initialize a variety of regression models to try.
        """
        logger.info("Initializing models...")
        
        # basic linear models
        self.models['linear'] = LinearRegression()
        self.models['ridge'] = Ridge(random_state=self.random_state)
        self.models['lasso'] = Lasso(random_state=self.random_state)
        self.models['elasticnet'] = ElasticNet(random_state=self.random_state)
        
        # tree-based models
        self.models['random_forest'] = RandomForestRegressor(random_state=self.random_state)
        self.models['gradient_boosting'] = GradientBoostingRegressor(random_state=self.random_state)
        
        # advanced gradient boosting libraries
        self.models['xgboost'] = xgb.XGBRegressor(random_state=self.random_state)
        self.models['lightgbm'] = lgb.LGBMRegressor(random_state=self.random_state, verbose=-1)
        self.models['catboost'] = cb.CatBoostRegressor(random_state=self.random_state, verbose=0)
        
        logger.info(f"Initialized {len(self.models)} models")

    def evaluate_model(self, model, X, y, cv=5):
        """
        Evaluate a model using cross-validation.
        
        Parameters:
        -----------
        model : estimator
            The model to evaluate
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series
            Target variable
        cv : int
            Number of cross-validation folds
        
        Returns:
        --------
        float
            Mean RMSE across CV folds
        """
        # calculate negative RMSE
        neg_rmse_scores = cross_val_score(
            model, X, y,
            scoring = 'neg_root_mean_squared_error',
            cv = cv,
            n_jobs = -1
        )

        # convert back to positive RMSE
        rmse_scores = -neg_rmse_scores

        mean_rmse = rmse_scores.mean()
        std_rmse = rmse_scores.std()

        return mean_rmse, std_rmse
    
    def initial_model_screening(self, X_train, y_train, cv=5):
        """
        Run a quick evaluation of all models to identify promising ones.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features
        y_train : pandas.Series
            Target variable
        cv : int
            Number of cross-validation folds
        
        Returns:
        --------
        dict
            Dictionary of model performance metrics
        """
        logger.info('Starting initial model screening...')

        results = {}

        for name, model in self.models.items():
            start_time = time.time()
            logger.info(f'Evaluating {name}')

            try:
                rmse, rmse_std = self.evaluate_model(model, X_train, y_train, cv)
                training_time = time.time() - start_time

                results[name] = {
                    'rmse': rmse,
                    'rmse_std': rmse_std,
                    'time': training_time
                }

                logger.info(f'{name} - RMSE; {rmse:.2f} ± {rmse_std:.2f}, Time: {training_time:.2f}s')

                # track best model so far
                if rmse < self.best_score:
                    self.best_score = rmse
                    self.best_model_name = name

            except Exception as e:
                logger.error(f'Error evaluating {name}: {str(e)}')

        logger.info(f'Initial screening complete. Best model: {self.best_model_name} with RMSE: {self.best_score:.2f}')
        return results
    
    def tune_hyperparameters(self, X_train, y_train, top_n=3):
        """
        Perform hyperparameter tuning on the best models.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features
        y_train : pandas.Series
            Target variable
        top_n : int
            Number of top models to tune
        """
        logger.info('Starting hyperparameter tuning...')

        # create a validation set for tuning
        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state
        )

        # sort models by performance
        sorted_models = sorted(
            [(name, self.models[name], results['rmse'])
             for name, results, in self.results.items()],
             key = lambda x: x[2]
        )

        # take top N models for tuning
        top_models = sorted_models[:top_n]
        logger.info(f'Tuning top {top_n} models: {[name for name, _, _ in top_models]}')

        for name, model, rmse in top_models:
            logger.info(f'Tuning {name}...')

            try:
                # define hyperparameters to tune based on model type
                if name == 'ridge':
                    param_grid = {
                        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
                        'fit_intercept': [True, False],
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                    }
                
                elif name == 'lasso':
                    param_grid = {
                        'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
                        'fit_intercept': [True, False],
                        'selection': ['cyclic', 'random'],
                        'max_iter': [1000, 2000, 3000, 5000]
                    }
                
                elif name == 'elasticnet':
                    param_grid = {
                        'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
                        'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        'fit_intercept': [True, False],
                        'selection': ['cyclic', 'random'],
                        'max_iter': [1000, 2000, 5000]
                    }
                
                elif name == 'random_forest':
                    param_grid = {
                        'n_estimators': [50, 100, 200, 300, 500],
                        'max_depth': [None, 5, 10, 15, 20, 30],
                        'min_samples_split': [2, 5, 10, 15],
                        'min_samples_leaf': [1, 2, 4, 8],
                        'max_features': ['auto', 'sqrt', 0.5, 0.7, 1.0]
                    }
                
                elif name == 'gradient_boosting':
                    param_grid = {
                        'n_estimators': [100, 200, 300, 500],
                        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
                        'max_depth': [3, 4, 5, 6, 8],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'subsample': [0.7, 0.8, 0.9, 1.0]
                    }
                
                elif name == 'xgboost':
                    param_grid = {
                        'n_estimators': [100, 200, 300, 500],
                        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
                        'max_depth': [3, 4, 5, 6, 8],
                        'min_child_weight': [1, 3, 5, 7],
                        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                        'gamma': [0, 0.1, 0.2, 0.3, 0.5],
                        'reg_alpha': [0, 0.1, 1.0, 10.0],
                        'reg_lambda': [0, 0.1, 1.0, 10.0]
                    }
                
                elif name == 'lightgbm':
                    param_grid = {
                        'n_estimators': [100, 200, 300, 500],
                        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
                        'max_depth': [3, 4, 5, 6, 8, -1],
                        'num_leaves': [31, 63, 127, 255],
                        'min_child_samples': [5, 10, 20, 50],
                        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                        'reg_alpha': [0, 0.1, 1.0, 10.0],
                        'reg_lambda': [0, 0.1, 1.0, 10.0]
                    }
                
                elif name == 'catboost':
                    param_grid = {
                        'iterations': [100, 200, 300, 500],
                        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
                        'depth': [4, 6, 8, 10],
                        'l2_leaf_reg': [1, 3, 5, 7, 9],
                        'border_count': [32, 64, 128, 254],
                        'bagging_temperature': [0, 1, 5, 10],
                        'random_strength': [0.1, 1, 10]
                    }
                
                else:
                    logger.info(f"No hyperparameter grid defined for {name}, skipping tuning")
                    continue

                # create grid search
                if len(param_grid) > 4:  # If grid is very large
                    from sklearn.model_selection import RandomizedSearchCV
                    search = RandomizedSearchCV(
                        model, param_grid,
                        n_iter=50,  # Number of parameter settings sampled
                        scoring='neg_root_mean_squared_error',
                        cv=5, n_jobs=-1, verbose=1,
                        random_state=self.random_state
                    )
                    logger.info(f"Using RandomizedSearchCV for {name} with 50 iterations")
                else:
                    search = GridSearchCV(
                        model, param_grid,
                        scoring='neg_root_mean_squared_error',
                        cv=5, n_jobs=-1, verbose=1
                    )
                    logger.info(f"Using GridSearchCV for {name}")

                # fit grid search
                search.fit(X_train_main, y_train_main)

                # update model with best parameters
                self.models[name] = search.best_estimator_

                # evaluate on validation set
                y_pred = self.models[name].predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))

                logger.info(f'{name} tuned - Best parameters: {search.best_params_}')
                logger.info(f'{name} tuned - Validation RMSE: {val_rmse:.2f}')

                # check if this is our new best model
                if val_rmse < self.best_score:
                    self.best_score = val_rmse
                    self.best_model_name = name
                    self.best_model = self.models[name]

            except Exception as e:
                logger.error(f'Error tuning {name}: {str(e)}')

        logger.info(f'Hyperparameter tuning complete. Best model: {self.best_model_name} with RMSE: {self.best_score:.2f}')

    def evaluate_final_model(self, X_train, y_train, cv=10):
        """
        Perform thorough evaluation of the best model.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features
        y_train : pandas.Series
            Target variable
        cv : int
            Number of cross-validation folds
        """
        logger.info(f"Evaluating final model: {self.best_model_name}")
        
        # define cross-validation strategy
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # arrays to store results
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        fold_importances = []
        
        # perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            # split data
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # train model
            model = self.models[self.best_model_name]
            model.fit(X_fold_train, y_fold_train)
            
            # make predictions
            y_pred = model.predict(X_fold_val)
            
            # calculate metrics
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            mae = mean_absolute_error(y_fold_val, y_pred)
            r2 = r2_score(y_fold_val, y_pred)
            
            # store metrics
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            
            # store feature importances if available
            if hasattr(model, 'feature_importances_'):
                fold_importances.append(
                    pd.Series(model.feature_importances_, index=X_train.columns)
                )
            
            logger.info(f"Fold {fold+1}/{cv} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        # calculate overall metrics
        mean_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)
        mean_mae = np.mean(mae_scores)
        mean_r2 = np.mean(r2_scores)
        
        logger.info(f"Final evaluation - RMSE: {mean_rmse:.2f} ± {std_rmse:.2f}")
        logger.info(f"Final evaluation - MAE: {mean_mae:.2f}")
        logger.info(f"Final evaluation - R²: {mean_r2:.4f}")
        
        # compute and plot feature importances if available
        if fold_importances:
            mean_importances = pd.concat(fold_importances, axis=1).mean(axis=1)
            self.plot_feature_importance(mean_importances)

    def plot_feature_importance(self, importances):
        """
        Plot feature importances.
        
        Parameters:
        -----------
        importances : pandas.Series
            Feature importances
        """
        plt.figure(figsize=(12, 8))
        
        # sort importances and plot
        importances = importances.sort_values(ascending=False)
        sns.barplot(x=importances.values, y=importances.index)
        
        plt.title(f'Feature Importances for {self.best_model_name}')
        plt.tight_layout()
        
        # save plot
        plt.savefig(os.path.join(project_root, 'reports', 'figures', f'{self.best_model_name}_feature_importance.png'))
        plt.savefig(f'../reports/figures/{self.best_model_name}_feature_importance.png')
        logger.info(f"Feature importance plot saved to ../reports/figures/{self.best_model_name}_feature_importance.png")

    def train_final_model(self, X_train, y_train):
        """
        Train the final model on all training data.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features
        y_train : pandas.Series
            Target variable
        """
        logger.info(f'Training final {self.best_model_name} model on all data')

        # get the best model
        model = self.models[self.best_model_name]

        # fit on all data
        model.fit(X_train, y_train)

        # save model
        model_path = os.path.join(self.models_dir, f'{self.best_model_name}_final.pkl')
        joblib.dump(model, model_path)
        logger.info(f'Final model saved to {model_path}')

        # save model information
        model_info = {
            'model_name': self.best_model_name,
            'model_params': model.get_params(),
            'rmse_cv': self.best_score,
            'features': list(X_train.columns)
        }

        with open(os.path.join(self.models_dir, 'model_info.pkl'), 'wb') as f:
            pickle.dump(model_info, f)

        # return the trained model
        return model
    
    def generate_submission(self, model, X_test, test_ids):
        """
        Generate submission file.
        
        Parameters:
        -----------
        model : estimator
            Trained model
        X_test : pandas.DataFrame
            Test features
        test_ids : pandas.Series
            Test registration numbers
        """
        logger.info('Generating submission file')

        # generate predictions
        y_pred = model.predict(X_test)

        # create submission dataframe
        submission = pd.DataFrame({
            'Registration Number': test_ids,
            'Annual Turnover': y_pred
        })

        # save submission
        submission_path = os.path.join(
            self.submission_dir,
            f"submission_{self.best_model_name}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        submission.to_csv(submission_path, index=False)

        logger.info(f'Submission saved to {submission_path}')
        logger.info(f"Submission predictions - Min: {y_pred.min():.2f}, Max: {y_pred.max():.2f}, Mean: {y_pred.mean():.2f}")

        return submission
    
    def run_pipeline(self, transform_target=True):
        """
        Run the full model training pipeline.
        
        Returns:
        --------
        pandas.DataFrame
            Submission file
        """
        # load data
        X_train, y_train, X_test, test_ids = self.load_data()

        # initialize feature processor if not already created
        if not hasattr(self, 'feature_processor'):
            from src.features.feature_processor import FeatureProcessor
            self.feature_processor = FeatureProcessor(target_column='Annual Turnover')

        # transform target if requested
        if transform_target:
            logger.info("Applying log transformation to target variable")
            self.y_train_original = y_train.copy()
            y_train = self.feature_processor.transform_target(y_train)

        # initialize models
        self.initialize_models()

        # evaluate all models
        self.results = self.initial_model_screening(X_train, y_train)

        # tune top models
        self.tune_hyperparameters(X_train, y_train)

        # evaluate final model
        self.evaluate_final_model(X_train, y_train)

        # train final model
        final_model = self.train_final_model(X_train, y_train)

        # generate submission
        if transform_target:
        # we need to inverse transform the predictions
            submission = self.generate_submission_with_transformed_target(
                final_model, X_test, test_ids
            )
        else:
            # Standard prediction
            submission = self.generate_submission(final_model, X_test, test_ids)

        logger.info('Model training pipeline completed successfully')
        return submission
    
    def generate_submission_with_transformed_target(self, model, X_test, test_ids):
        """
        Generate submission file with inverse transformation of target.
    
        Parameters:
        -----------
        model : estimator
            Trained model
        X_test : pandas.DataFrame
            Test features
        test_ids : pandas.Series
            Test registration numbers
        """
        logger.info('Generating submission file with inverse target transformation')

        # generate log-transformed predictions
        y_pred_log = model.predict(X_test)
    
        # inverse transform to original scale
        y_pred = self.feature_processor.inverse_transform_target(y_pred_log)

        # create submission dataframe
        submission = pd.DataFrame({
            'Registration Number': test_ids,
            'Annual Turnover': y_pred
        })

        # save submission
        submission_path = os.path.join(
            self.submission_dir,
            f"submission_{self.best_model_name}_log_transform_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        submission.to_csv(submission_path, index=False)

        logger.info(f'Submission saved to {submission_path}')
        logger.info(f"Submission predictions - Min: {y_pred.min():.2f}, Max: {y_pred.max():.2f}, Mean: {y_pred.mean():.2f}")

        return submission
    
if __name__ == "__main__":
    # run the model training pipeline
    trainer = ModelTrainer()
    submission = trainer.run_pipeline()