# src/models/ensemble_trainer.py

import os
import logging
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import pickle
import sys

# add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# import custom modules
from src.models.train_model import ModelTrainer
from src.models.ensemble import StackingEnsemble, VotingEnsemble, BlendingEnsemble, optimize_ensemble_weights

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs/ensemble_training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ensemble_training')

class EnsembleTrainer(ModelTrainer):
    """
    A class to train and evaluate ensemble models for restaurant turnover prediction.
    """

    def __init__(self, random_state=42):
        """
        Initialize the ensemble trainer.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        super().__init__(random_state=random_state)
        self.ensemble_models = {}
        self.best_ensemble = None
        self.best_ensemble_name = None
        self.best_ensemble_score = float('inf')

    def create_ensemble_models(self):
        """
        Create various ensemble models to evaluate.
        """
        logger.info('Creating ensemble models')

        # 1. create base models
        base_models = [
            ('ridge', Ridge(alpha=10.0, fit_intercept=False, random_state=self.random_state)),
            ('lasso', Lasso(alpha=1.0, random_state=self.random_state)),
            ('elastic_net', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state)),
            ('gradient_boosting', GradientBoostingRegressor(
                learning_rate=0.05, max_depth=3, n_estimators=200, 
                subsample=0.9, random_state=self.random_state
            )),
            ('random_forest', RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5, 
                min_samples_leaf=2, random_state=self.random_state
            )),
            ('xgboost', xgb.XGBRegressor(
                learning_rate=0.05, max_depth=4, n_estimators=200,
                subsample=0.8, colsample_bytree=0.8, random_state=self.random_state
            )),
            ('lightgbm', lgb.LGBMRegressor(
                learning_rate=0.05, max_depth=5, n_estimators=200,
                subsample=0.8, colsample_bytree=0.8, random_state=self.random_state,
                verbose=-1
            ))
        ]

        # 2. create meta-learners
        meta_learners = {
            'ridge': Ridge(alpha=10.0, random_state=self.random_state),
            'lasso': Lasso(alpha=0.01, random_state=self.random_state),
            'gbm': GradientBoostingRegressor(
                learning_rate=0.05, max_depth=3, n_estimators=100, 
                subsample=0.8, random_state=self.random_state
            )
        }

        # 3. create ensemble models

        # stacking ensembles
        for meta_name, meta_model in meta_learners.items():
            # basic stacking
            self.ensemble_models[f'stack_{meta_name}'] = StackingEnsemble(
                base_models=base_models,
                meta_model=meta_model,
                n_folds=5,
                use_features_in_meta=False,
                random_state=self.random_state
            )

            # stacking with original features
            self.ensemble_models[f'stack_{meta_name}_with_features'] = StackingEnsemble(
                base_models=base_models,
                meta_model=meta_model,
                n_folds=5,
                use_features_in_meta=True,
                random_state=self.random_state
            )

        # blending ensembles
        for meta_name, meta_model in meta_learners.items():
            self.ensemble_models[f'blend_{meta_name}'] = BlendingEnsemble(
                base_models=base_models,
                meta_model=meta_model,
                val_size=0.2,
                random_state=self.random_state,
                use_features_in_meta=False
            )

        # voting ensemble
        self.ensemble_models['voting_equal'] = VotingEnsemble(
            models=base_models,
            weights=None    # equal weights
        )

        logger.info(f'Created {len(self.ensemble_models)} ensemble models')

    def evaluate_ensemble_models(self, X_train, y_train, cv=5):
        """
        Evaluate ensemble models using cross-validation.
        
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
        logger.info('Starting ensemble model evaluation')

        # create a validation set for final evaluation
        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state
        )

        results = {}

        for name, model in self.ensemble_models.items():
            start_time = time.time()
            logger.info(f'Evaluating {name}')

            try:
                # fit model on training data
                model.fit(X_train_main, y_train_main)

                # evaluate on validation set
                y_pred = model.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                val_r2 = r2_score(y_val, y_pred)

                # store results
                training_time = time.time() - start_time
                results[name] = {
                    'rmse': val_rmse,
                    'r2': val_r2,
                    'time': training_time
                }

                logger.info(f'{name} - RMSE: {val_rmse:.2f}, R2: {val_r2:.4f}. Time: {training_time:.2f}s')

                # track best model
                if val_rmse < self.best_ensemble_score:
                    self.best_ensemble_score = val_rmse
                    self.best_ensemble_name = name
                    self.best_ensemble = model

            except Exception as e:
                logger.error(f'Error evaluating {name}: {str(e)}')

        logger.info(f'Ensemble evaluating complete. Best model: {self.best_ensemble_name} with RMSE: {self.best_ensemble_score:.2f}')
        return results
    
    def optimize_voting_ensemble(self, X_train, y_train):
        """
        Optimize weights for the voting ensemble.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features
        y_train : pandas.Series
            Target variable
        
        Returns:
        --------
        VotingEnsemble
            Optimized voting ensemble
        """
        logger.info('Optimizing voting ensemble weights')

        # create a basic voting ensemble if it doesn't exist
        if 'voting_equal' not in self.ensemble_models:
            base_models = [
                ('ridge', Ridge(alpha=0.1, fit_intercept=False, random_state=self.random_state)),
                ('lasso', Lasso(alpha=1.0, random_state=self.random_state)),
                ('elastic_net', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state)),
                ('gradient_boosting', GradientBoostingRegressor(
                    learning_rate=0.05, max_depth=3, n_estimators=200, 
                    subsample=0.9, random_state=self.random_state
                )),
                ('random_forest', RandomForestRegressor(
                    n_estimators=200, max_depth=15, min_samples_split=5, 
                    min_samples_leaf=2, random_state=self.random_state
                )),
                ('xgboost', xgb.XGBRegressor(
                    learning_rate=0.05, max_depth=4, n_estimators=200,
                    subsample=0.8, colsample_bytree=0.8, random_state=self.random_state
                )),
                ('lightgbm', lgb.LGBMRegressor(
                    learning_rate=0.05, max_depth=5, n_estimators=200,
                    subsample=0.8, colsample_bytree=0.8, random_state=self.random_state,
                    verbose=-1
                ))
            ]
            
            self.ensemble_models['voting_equal'] = VotingEnsemble(
                models=base_models,
                weights=None
            )
        
        # get the voting ensemble
        voting_ensemble = self.ensemble_models['voting_equal']

        # optimize weights
        best_weights = optimize_ensemble_weights(
            ensemble=voting_ensemble,
            X=X_train,
            y=y_train,
            n_trials=50,  # Increase for better results
            cv=5,
            random_state=self.random_state
        )
        
        # create optimized voting ensemble
        optimized_ensemble = VotingEnsemble(
            models=voting_ensemble.models,
            weights=best_weights
        )

        # add to ensemble models
        self.ensemble_models['voting_optimized'] = optimized_ensemble
        
        # fit on all data
        optimized_ensemble.fit(X_train, y_train)
        
        # evaluate
        cv_scores = cross_val_score(
            optimized_ensemble, X_train, y_train,
            scoring='neg_root_mean_squared_error',
            cv=5, n_jobs=-1
        )
        rmse = -np.mean(cv_scores)
        
        logger.info(f'Optimized voting ensemble - RMSE: {rmse:.2f}')
        
        # check if this is better than our current best
        if rmse < self.best_ensemble_score:
            self.best_ensemble_score = rmse
            self.best_ensemble_name = 'voting_optimized'
            self.best_ensemble = optimized_ensemble
            logger.info('Optimized voting ensemble is the new best model')
        
        return optimized_ensemble
    
    def train_final_ensemble(self, X_train, y_train):
        """
        Train the final ensemble model on all training data.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features
        y_train : pandas.Series
            Target variable
        
        Returns:
        --------
        object
            Trained ensemble model
        """
        logger.info(f'Training final {self.best_ensemble_name} ensemble on all data')

        # train the best ensemble on all data
        start_time = time.time()
        self.best_ensemble.fit(X_train, y_train)
        training_time = time.time() - start_time

        logger.info(f'Ensemble training completed in {training_time:.2f}s')
        
        # save model
        model_path = os.path.join(self.models_dir, f'ensemble_{self.best_ensemble_name}_final.pkl')
        joblib.dump(self.best_ensemble, model_path)
        logger.info(f'Final ensemble model saved to {model_path}')
        
        # save model information
        model_info = {
            'model_name': f'ensemble_{self.best_ensemble_name}',
            'model_type': self.best_ensemble_name,
            'rmse_cv': self.best_ensemble_score,
            'training_time': training_time
        }
        
        with open(os.path.join(self.models_dir, 'ensemble_info.pkl'), 'wb') as f:
            pickle.dump(model_info, f)
        
        return self.best_ensemble
    
    def generate_ensemble_submission(self, model, X_test, test_ids, 
                                transform_target=True, suffix="ensemble"):
        """
        Generate a submission file from ensemble model predictions.
        
        Parameters:
        -----------
        model : estimator
            Trained ensemble model
        X_test : pandas.DataFrame
            Test features
        test_ids : pandas.Series
            Test registration numbers
        transform_target : bool
            Whether target was log-transformed and needs inverse transformation
        suffix : str
            Suffix to add to the filename
            
        Returns:
        --------
        pandas.DataFrame
            Submission dataframe
        """
        logger.info("Generating submission file for ensemble model")
        
        try:
            # Generate predictions
            y_pred = model.predict(X_test)
            
            # Apply inverse transformation if needed
            if transform_target:
                # Check if we have a feature processor
                if not hasattr(self, 'feature_processor') or self.feature_processor is None:
                    # Create a feature processor if not already created
                    logger.warning("Feature processor not found, creating new one")
                    from src.features.feature_processor import FeatureProcessor
                    self.feature_processor = FeatureProcessor(target_column='Annual Turnover')
                
                logger.info("Applying inverse transformation to predictions")
                try:
                    y_pred = self.feature_processor.inverse_transform_target(y_pred)
                except Exception as e:
                    logger.error(f"Error in inverse transformation: {str(e)}")
                    # Fallback: apply expm1 manually
                    logger.info("Falling back to manual inverse transformation")
                    y_pred = np.expm1(y_pred)
            
            # Create submission dataframe
            submission = pd.DataFrame({
                'Registration Number': test_ids,
                'Annual Turnover': y_pred
            })
            
            # Generate timestamp
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            
            # Create submission path
            model_name = self.best_ensemble_name if hasattr(self, 'best_ensemble_name') else "ensemble"
            submission_path = os.path.join(
                self.submission_dir,
                f"submission_{model_name}_{suffix}_{timestamp}.csv"
            )
            
            # Save submission
            submission.to_csv(submission_path, index=False)
            
            # Log statistics
            logger.info(f"Submission saved to {submission_path}")
            logger.info(f"Prediction statistics: Min: {y_pred.min():.2f}, Max: {y_pred.max():.2f}, Mean: {y_pred.mean():.2f}")
            
            return submission
        
        except Exception as e:
            logger.error(f"Error generating submission: {str(e)}")
            # Create a basic submission as fallback
            logger.warning("Creating fallback submission")
            submission = pd.DataFrame({
                'Registration Number': test_ids,
                'Annual Turnover': np.ones(len(test_ids)) * 30000000  # Fallback value based on mean
            })
            
            # Save fallback submission
            fallback_path = os.path.join(
                self.submission_dir,
                f"fallback_submission_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            )
            submission.to_csv(fallback_path, index=False)
            logger.info(f"Fallback submission saved to {fallback_path}")
            
            return submission
    
    def run_ensemble_pipeline(self, transform_target=True):
        """
        Run the full ensemble model training pipeline.
        
        Parameters:
        -----------
        transform_target : bool
            Whether to apply log transformation to target variable
            
        Returns:
        --------
        pandas.DataFrame
            Submission file
        """
        # load data
        X_train, y_train, X_test, test_ids = self.load_data()

        # transform target if requested
        if transform_target:
            logger.info('Applying log transformation to target variable')
            from src.features.feature_processor import FeatureProcessor
            self.feature_processor = FeatureProcessor(target_column='Annial Turnover')
            self.y_train_original = y_train.copy()
            y_train = self.feature_processor.transform_target(y_train)

        from sklearn.feature_selection import SelectKBest, f_regression

        # select top 25 features to reduce multicollinearity
        logger.info('Applying feature selection')
        selector = SelectKBest(f_regression, k=25)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_train_selected = pd.DataFrame(X_train_selected, columns=X_train.columns[selector.get_support()])
        X_test_selected = selector.transform(X_test)
        X_test_selected = pd.DataFrame(X_test_selected, columns=X_train.columns[selector.get_support()])

        X_train = X_train_selected
        X_test = X_test_selected

        # create ensemble models
        self.create_ensemble_models()

        # evaluate ensemble models
        self.ensemble_results = self.evaluate_ensemble_models(X_train, y_train)

        # optimize voting ensemble
        self.optimize_voting_ensemble(X_train, y_train)

        # train final ensemble
        final_ensemble = self.train_final_ensemble(X_train, y_train)

        # generate submission
        submission = self.generate_ensemble_submission(
            final_ensemble, X_test, test_ids, transform_target=transform_target
        )

        logger.info('Ensemble training pipeline completed successfully')
        return submission
    
if __name__ == "__main__":
    # run ensemble training pipeline
    trainer = EnsembleTrainer()
    submission = trainer.run_ensemble_pipeline()