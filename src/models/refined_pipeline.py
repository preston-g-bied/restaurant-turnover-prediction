import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.base import clone
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import time
import os
import scipy.stats as stats
import logging
import sys
from datetime import datetime
from scipy import special

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

class RefinedPipeline:
    def __init__(self, random_state=42, log_level=logging.INFO):
        self.random_state = random_state
        
        # Set up logging
        self.setup_logging(log_level)
        
    def setup_logging(self, log_level):
        """Configure logging for the pipeline"""
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(project_root, 'logs')
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            
        # Configure the root logger
        self.logger = logging.getLogger('RefinedPipeline')
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Create a file handler with timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f'refined_pipeline_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        
        # Create a console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Create a formatter and set it for both handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging configured. Log file: {log_file}")
        
    def run(self, data_path=os.path.join(project_root, 'data', 'processed')):
        """Execute the full pipeline"""
        self.logger.info("Starting refined pipeline...")
        start_time = time.time()
        
        try:
            # 1. Load data
            self.logger.info(f"Loading data from {data_path}")
            train_data = pd.read_csv(os.path.join(data_path, 'train_processed.csv'))
            test_data = pd.read_csv(os.path.join(data_path, 'test_processed.csv'))
            self.logger.info(f"Loaded training data: {train_data.shape}, test data: {test_data.shape}")
            
            # 2. Extract target and test IDs
            self.logger.info("Extracting target variable and test IDs")
            y_train = train_data['Annual Turnover'] 
            X_train = train_data.drop('Annual Turnover', axis=1)
            test_ids = test_data['Registration Number']
            X_test = test_data.drop('Registration Number', axis=1)
            
            # 3. Handle outliers
            self.logger.info("Handling outliers in target variable")
            X_train, y_train = self.handle_outliers(X_train, y_train)
            
            # 4. Try different target transformations
            self.logger.info("Exploring target variable transformations")
            y_train_transformed, transform_details = self.try_different_transformations(y_train)
            self.logger.info(f"Selected transformation: {transform_details['name']}")
            
            # Make sure y_train_transformed is a pandas Series with the same index as X_train
            if isinstance(y_train_transformed, np.ndarray):
                self.logger.info("Converting transformed target array to pandas Series")
                y_train_transformed = pd.Series(y_train_transformed, index=X_train.index)

            # 5. Select features using domain knowledge and importance
            self.logger.info("Selecting features with domain knowledge")
            selected_features = self.select_features_with_domain_knowledge(X_train, y_train_transformed)
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            self.logger.info(f"Selected {len(selected_features)} features")
            
            # 6. Add carefully chosen interaction features
            self.logger.info("Adding targeted interaction features")
            X_train_enhanced, X_test_enhanced = self.add_targeted_interactions(
                X_train_selected, X_test_selected, selected_features
            )
            self.logger.info(f"Feature dimensions after interaction: train={X_train_enhanced.shape}, test={X_test_enhanced.shape}")
            
            # 7. Standardize features
            self.logger.info("Standardizing features")
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train_enhanced),
                columns=X_train_enhanced.columns
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test_enhanced),
                columns=X_test_enhanced.columns
            )
            
            # 8. Optimize model hyperparameters
            self.logger.info("Starting hyperparameter optimization")
            optimized_models = self.optimize_hyperparameters(X_train_scaled, y_train_transformed)
            self.logger.info(f"Optimized {len(optimized_models)} models")
            
            # 9. Generate enhanced stacked predictions
            self.logger.info("Generating stacked predictions")
            final_preds_transformed = self.enhanced_stacking(
                X_train_scaled, y_train_transformed, X_test_scaled, optimized_models
            )
            
            # 10. Inverse transform predictions with safeguards
            self.logger.info(f"Applying inverse {transform_details['name']} transformation to predictions")
            if transform_details['name'] == 'log':
                final_preds = np.expm1(final_preds_transformed)
                self.logger.debug(f"Applied np.expm1 transform")
            elif transform_details['name'] == 'boxcox':
                final_preds = special.inv_boxcox(final_preds_transformed, transform_details['lambda'])
                self.logger.debug(f"Applied special.inv_boxcox transform with lambda={transform_details['lambda']}")
            elif transform_details['name'] == 'yeojohnson':
                final_preds = stats.inv_yeojohnson(final_preds_transformed, transform_details['lambda'])
                self.logger.debug(f"Applied inv_yeojohnson transform with lambda={transform_details['lambda']}")
            elif transform_details['name'] == 'sqrt':
                final_preds = np.square(final_preds_transformed)
                self.logger.debug(f"Applied square transform")
            
            # 11. Apply range clipping based on training data
            original_min, original_max = y_train.min(), y_train.max()
            self.logger.info(f"Clipping predictions to range [{original_min * 0.9:.2f}, {original_max * 1.1:.2f}]")
            final_preds = np.clip(
                final_preds,
                original_min * 0.9,  # Allow 10% below minimum 
                original_max * 1.1   # Allow 10% above maximum
            )
            
            # Log prediction statistics before submission
            self.logger.info(f"Prediction stats - Min: {final_preds.min():.2f}, Max: {final_preds.max():.2f}, Mean: {final_preds.mean():.2f}")
            
            # 12. Create submission
            self.logger.info("Creating submission dataframe")
            submission = pd.DataFrame({
                'Registration Number': test_ids,
                'Annual Turnover': final_preds
            })
            
            # 13. Save submission with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            submission_path = os.path.join(project_root, 'data', 'submissions', f'refined_submission_{timestamp}.csv')
            submission.to_csv(submission_path, index=False)
            
            self.logger.info(f"Submission saved to {submission_path}")
            
            # Log time taken
            execution_time = time.time() - start_time
            self.logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
            
            return submission
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            raise
    
    def extract_feature_importance(self, X_train, y_train_log):
        """Extract feature importance from multiple models"""
        self.logger.info("Extracting feature importance from multiple models")
        
        # Train models for feature importance
        feature_models = {
            'xgb': xgb.XGBRegressor(n_estimators=100, random_state=self.random_state),
            'lgb': lgb.LGBMRegressor(n_estimators=100, random_state=self.random_state, verbose=-1)
        }
        
        importances = {}
        for name, model in feature_models.items():
            self.logger.debug(f"Training {name} for feature importance")
            model.fit(X_train, y_train_log)
            if hasattr(model, 'feature_importances_'):
                imp = pd.Series(model.feature_importances_, index=X_train.columns)
                importances[name] = imp
                self.logger.debug(f"Top 5 important features from {name}: {imp.sort_values(ascending=False).head(5).to_dict()}")
        
        # Combine importance scores from different models
        combined_imp = pd.DataFrame(importances)
        mean_imp = combined_imp.mean(axis=1).sort_values(ascending=False)
        
        # Select top features (top 20)
        top_features = mean_imp.head(20).index.tolist()
        self.logger.info(f"Top 20 features selected: {', '.join(top_features[:5])}...")
        
        return top_features, mean_imp
    
    def add_targeted_interactions(self, X_train, X_test, top_features):
        """Add targeted interaction features based on domain knowledge and feature importance"""
        self.logger.info("Adding targeted interaction features")
        
        X_train_enhanced = X_train.copy()
        X_test_enhanced = X_test.copy()
        
        # First, identify the most important features based on correlation with target
        important_features = []
        
        # These are the most important features for restaurant turnover based on domain knowledge
        key_features = [
            'Hygiene Rating',
            'Food Rating', 
            'Restaurant Zomato Rating',
            'Overall Restaurant Rating',
            'Facebook Popularity Quotient',
            'Instagram Popularity Quotient',
            'restaurant_age_years'
        ]
        
        # Filter to only include features that exist in the dataset
        important_features = [f for f in key_features if f in X_train.columns]
        self.logger.debug(f"Important features found in data: {important_features}")
        
        # Define a very small set of high-value interactions
        interaction_pairs = [
            # Quality rating interactions (highest importance)
            ('Hygiene Rating', 'Food Rating'),
            ('Restaurant Zomato Rating', 'Overall Restaurant Rating'),
            
            # Social media impact (second highest importance) 
            ('Facebook Popularity Quotient', 'Instagram Popularity Quotient')
        ]
        
        # Only create interactions if both features are available
        created_features = []
        
        # Add multiplicative interactions
        for feat1, feat2 in interaction_pairs:
            if feat1 in X_train.columns and feat2 in X_train.columns:
                # Multiplication interaction
                feature_name = f'{feat1}_{feat2}_mult'
                X_train_enhanced[feature_name] = X_train[feat1] * X_train[feat2]
                X_test_enhanced[feature_name] = X_test[feat1] * X_test[feat2]
                created_features.append(feature_name)
                self.logger.debug(f"Created interaction feature: {feature_name}")
        
        # Create up to 3 squared terms for the most important features
        for i, feature in enumerate(important_features):
            if i < 3 and feature in X_train.columns:  # Limit to top 3 features
                feat_name = f'{feature}_squared'
                X_train_enhanced[feat_name] = X_train[feature] ** 2
                X_test_enhanced[feat_name] = X_test[feature] ** 2
                created_features.append(feat_name)
                self.logger.debug(f"Created squared feature: {feat_name}")
        
        self.logger.info(f"Created {len(created_features)} interaction features")
        return X_train_enhanced, X_test_enhanced
    
    def select_features_with_domain_knowledge(self, X_train, y_train_log, importance_threshold=0.01):
        """Select features based on both importance and domain knowledge"""
        self.logger.info("Selecting features using domain knowledge and feature importance")
        
        # IMPORTANT FIX: Convert numpy array to pandas Series if needed
        if isinstance(y_train_log, np.ndarray):
            y_train_log = pd.Series(y_train_log, index=X_train.index)
            self.logger.info("Converted numpy array target to pandas Series")
        
        # Calculate correlations with target
        correlations = X_train.apply(lambda x: x.corr(y_train_log) 
                                    if x.dtype.kind in 'ifc' else 0)
        
        # Log top correlations
        top_corr = correlations.abs().sort_values(ascending=False).head(10)
        self.logger.debug(f"Top 10 features by correlation: {top_corr.to_dict()}")
        
        # These are must-have features based on domain knowledge
        must_have_features = [
            'Hygiene Rating',
            'Facebook Popularity Quotient',
            'Instagram Popularity Quotient',
            'Restaurant Zomato Rating',
            'Overall Restaurant Rating',
            'Food Rating',
            'restaurant_age_years',
            'Staff Responsivness',
            'Value for Money'
        ]
        
        # Get features from importance analysis
        self.logger.info("Training models to determine feature importance")
        feature_models = {
            'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lgb': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        }
        
        importances = {}
        for name, model in feature_models.items():
            try:
                model.fit(X_train, y_train_log)
                if hasattr(model, 'feature_importances_'):
                    imp = pd.Series(model.feature_importances_, index=X_train.columns)
                    importances[name] = imp
                    self.logger.debug(f"Extracted feature importance from {name}")
            except Exception as e:
                self.logger.warning(f"Error extracting importance from {name}: {str(e)}")
        
        # Combine importance scores
        combined_imp = pd.DataFrame(importances)
        mean_imp = combined_imp.mean(axis=1)
        
        # Log top important features
        top_imp = mean_imp.sort_values(ascending=False).head(10)
        self.logger.debug(f"Top 10 features by importance: {top_imp.to_dict()}")
        
        # Select features above threshold
        important_features = mean_imp[mean_imp > importance_threshold].index.tolist()
        self.logger.info(f"Selected {len(important_features)} features by importance threshold")
        
        # Combine must-have and important features
        selected_features = list(set(must_have_features) | set(important_features))
        
        # Keep only features that exist in the dataset
        final_features = [f for f in selected_features if f in X_train.columns]
        
        self.logger.info(f"Selected {len(final_features)} features using domain knowledge and importance")
        self.logger.debug(f"Selected features: {final_features}")
        return final_features
    
    def optimize_hyperparameters(self, X_train, y_train_log):
        """Optimize hyperparameters for key models using Bayesian optimization"""
        self.logger.info("Starting hyperparameter optimization with Optuna")
        
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            self.logger.error("Optuna not found. Install with: pip install optuna")
            raise ImportError("Optuna is required for hyperparameter optimization")
        
        optimized_models = {}
        
        # 1. Optimize Lasso with better convergence settings
        self.logger.info("Optimizing Lasso model")
        
        def lasso_objective(trial):
            params = {
                'alpha': trial.suggest_float('alpha', 0.0001, 1.0, log=True),
                'max_iter': trial.suggest_categorical('max_iter', [50000, 100000]),
                'tol': trial.suggest_float('tol', 1e-6, 1e-4, log=True),
                'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'random_state': self.random_state
            }
            
            model = Lasso(**params)
            
            # Use cross-validation to evaluate
            kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, val_idx in kf.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                y_pred = model.predict(X_fold_val)
                scores.append(np.sqrt(mean_squared_error(y_fold_val, y_pred)))
            
            return np.mean(scores)
        
        # Create and run Lasso optimization
        try:
            lasso_study = optuna.create_study(direction='minimize', 
                                            sampler=TPESampler(seed=self.random_state))
            lasso_study.optimize(lasso_objective, n_trials=20)
            
            # Get best parameters and create model
            lasso_params = lasso_study.best_params
            optimized_models['lasso'] = Lasso(**lasso_params, random_state=self.random_state)
            self.logger.info(f"Optimized Lasso - params: {lasso_params}")
            self.logger.info(f"Lasso best RMSE: {lasso_study.best_value:.5f}")
        except Exception as e:
            self.logger.error(f"Error optimizing Lasso: {str(e)}")
        
        # 2. Optimize LightGBM
        self.logger.info("Optimizing LightGBM model")
        
        def lgb_objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'random_state': self.random_state,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            
            # Use cross-validation to evaluate
            kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, val_idx in kf.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
                
                # Fit without eval_set and early_stopping_rounds
                model.fit(X_fold_train, y_fold_train)
                
                # Predict and calculate score
                y_pred = model.predict(X_fold_val)
                scores.append(np.sqrt(mean_squared_error(y_fold_val, y_pred)))
            
            return np.mean(scores)
        
        # Create and run LightGBM optimization
        try:
            lgb_study = optuna.create_study(direction='minimize', 
                                        sampler=TPESampler(seed=self.random_state))
            lgb_study.optimize(lgb_objective, n_trials=25)
            
            # Get best parameters and create model
            lgb_params = lgb_study.best_params
            optimized_models['lgb'] = lgb.LGBMRegressor(**lgb_params, random_state=self.random_state)
            self.logger.info(f"Optimized LightGBM - params: {lgb_params}")
            self.logger.info(f"LightGBM best RMSE: {lgb_study.best_value:.5f}")
        except Exception as e:
            self.logger.error(f"Error optimizing LightGBM: {str(e)}")
        
        # 3. Optimize XGBoost
        self.logger.info("Optimizing XGBoost model")
        
        def xgb_objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'random_state': self.random_state
            }
            
            model = xgb.XGBRegressor(**params)
            
            # Use cross-validation to evaluate
            kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, val_idx in kf.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
                
                # Simple fit without eval_set
                model.fit(X_fold_train, y_fold_train)
                
                # Predict and calculate score
                y_pred = model.predict(X_fold_val)
                scores.append(np.sqrt(mean_squared_error(y_fold_val, y_pred)))
            
            return np.mean(scores)
        
        # Create and run XGBoost optimization
        try:
            xgb_study = optuna.create_study(direction='minimize', 
                                        sampler=TPESampler(seed=self.random_state))
            xgb_study.optimize(xgb_objective, n_trials=25)
            
            # Get best parameters and create model
            xgb_params = xgb_study.best_params
            optimized_models['xgb'] = xgb.XGBRegressor(**xgb_params, random_state=self.random_state)
            self.logger.info(f"Optimized XGBoost - params: {xgb_params}")
            self.logger.info(f"XGBoost best RMSE: {xgb_study.best_value:.5f}")
        except Exception as e:
            self.logger.error(f"Error optimizing XGBoost: {str(e)}")
        
        # 4. Add CatBoost to the ensemble as a new model type
        try:
            import catboost as cb
            self.logger.info("Optimizing CatBoost model")
            
            def catboost_objective(trial):
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
                    'iterations': trial.suggest_int('iterations', 200, 1000),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                    'random_strength': trial.suggest_float('random_strength', 1e-8, 10, log=True),
                    'random_seed': self.random_state,
                    'verbose': False
                }
                
                model = cb.CatBoostRegressor(**params)
                
                # Use cross-validation to evaluate
                kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
                scores = []
                
                for train_idx, val_idx in kf.split(X_train):
                    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_fold_train, y_fold_val = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
                    
                    model.fit(X_fold_train, y_fold_train,
                            eval_set=[(X_fold_val, y_fold_val)],
                            early_stopping_rounds=50,
                            verbose=False)
                    
                    # Use best iteration to predict
                    y_pred = model.predict(X_fold_val)
                    scores.append(np.sqrt(mean_squared_error(y_fold_val, y_pred)))
                
                return np.mean(scores)
            
            # Create and run CatBoost optimization
            cb_study = optuna.create_study(direction='minimize', 
                                            sampler=TPESampler(seed=self.random_state))
            cb_study.optimize(catboost_objective, n_trials=20)
            
            # Get best parameters and create model
            cb_params = cb_study.best_params
            optimized_models['catboost'] = cb.CatBoostRegressor(**cb_params, random_seed=self.random_state)
            self.logger.info(f"Optimized CatBoost - params: {cb_params}")
            self.logger.info(f"CatBoost best RMSE: {cb_study.best_value:.5f}")
        except ImportError:
            self.logger.warning("CatBoost not available, skipping...")
        except Exception as e:
            self.logger.error(f"Error optimizing CatBoost: {str(e)}")
        
        return optimized_models
    
    def stacked_predictions(self, X_train, y_train_log, X_test, models, random_state=42):
        """Generate stacked predictions using a meta-learner"""
        self.logger.info("Generating stacked predictions with cross-validation")
        
        # Convert to numpy for faster operations
        X_train_values = X_train.values
        X_test_values = X_test.values
        
        # Cross-validation setup
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        meta_train = np.zeros((len(X_train), len(models)))
        meta_test = np.zeros((len(X_test), len(models)))
        
        # Generate out-of-fold predictions for meta-training
        for i, (name, model) in enumerate(models.items()):
            self.logger.info(f"Generating meta-features for {name}...")
            test_preds = np.zeros(len(X_test))
            
            try:
                if name in ['lgb', 'xgb']:
                    # Cross-validation predictions
                    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                        X_fold_train = pd.DataFrame(X_train.iloc[train_idx], columns=X_train.columns)
                        X_fold_val = pd.DataFrame(X_train.iloc[val_idx], columns=X_train.columns)
                        y_fold_train = y_train_log.iloc[train_idx]
                        
                        # Train on fold
                        model_clone = clone(model)
                        model_clone.fit(X_fold_train, y_fold_train)
                        
                        # Predict on validation fold
                        meta_train[val_idx, i] = model_clone.predict(X_fold_val)
                        
                        # Test predictions
                        X_test_df = pd.DataFrame(X_test, columns=X_train.columns)
                        test_preds += model_clone.predict(X_test_df) / kf.n_splits
                        self.logger.debug(f"Completed fold {fold+1}/{kf.n_splits} for {name}")
                else:
                    # Cross-validation predictions
                    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_values)):
                        X_fold_train, X_fold_val = X_train_values[train_idx], X_train_values[val_idx]
                        y_fold_train = y_train_log.iloc[train_idx]
                        
                        # Train on fold
                        model_clone = clone(model)
                        model_clone.fit(X_fold_train, y_fold_train)
                        
                        # Predict on validation fold
                        meta_train[val_idx, i] = model_clone.predict(X_fold_val)
                        
                        # Contribute to test predictions
                        test_preds += model_clone.predict(X_test_values) / kf.n_splits
                        self.logger.debug(f"Completed fold {fold+1}/{kf.n_splits} for {name}")
                
                # Store test predictions
                meta_test[:, i] = test_preds
                
                # Also fit on full dataset
                model.fit(X_train_values, y_train_log)
                self.logger.info(f"Generated meta-features for {name} successfully")
            except Exception as e:
                self.logger.error(f"Error generating meta-features for {name}: {str(e)}")
                # Fill with zeros if a model fails
                meta_test[:, i] = 0.0
        
        # Create meta-learner (simple Ridge regression)
        self.logger.info("Training meta-learner...")
        meta_model = Ridge(alpha=0.5, random_state=random_state)
        
        # Train meta-learner
        meta_model.fit(meta_train, y_train_log)
        
        # Output meta-model coefficients (weights for each model)
        model_weights = {name: weight for name, weight in zip(models.keys(), meta_model.coef_)}
        self.logger.info(f"Meta-model weights: {model_weights}")
        
        # Make final predictions
        final_preds = meta_model.predict(meta_test)
        
        return final_preds
    
    def enhanced_stacking(self, X_train, y_train_log, X_test, models, random_state=42):
        """Create enhanced stacked predictions with model selection"""
        self.logger.info("Creating enhanced stacked predictions with model selection")
        
        # First evaluate each model individually with cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        model_rmse = {}
        
        for name, model in models.items():
            self.logger.info(f"Cross-validating {name} model")
            cv_scores = []
            try:
                for train_idx, val_idx in kf.split(X_train):
                    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_fold_train, y_fold_val = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
                    
                    model_clone = clone(model)
                    model_clone.fit(X_fold_train, y_fold_train)
                    y_pred = model_clone.predict(X_fold_val)
                    
                    rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
                    cv_scores.append(rmse)
                
                model_rmse[name] = np.mean(cv_scores)
                self.logger.info(f"Model {name} - CV RMSE: {model_rmse[name]:.4f}")
            except Exception as e:
                self.logger.error(f"Error evaluating {name}: {str(e)}")
        
        # Keep only the 3 best models for the final ensemble
        if model_rmse:
            top_models = sorted(model_rmse.items(), key=lambda x: x[1])[:3]
            top_model_names = [name for name, _ in top_models]
            
            self.logger.info(f"Using top {len(top_model_names)} models for ensemble: {top_model_names}")
            
            # Filter models
            selected_models = {name: models[name] for name in top_model_names}
            
            # Now create meta-features using only these models
            meta_train = np.zeros((len(X_train), len(selected_models)))
            meta_test = np.zeros((len(X_test), len(selected_models)))
            
            # Generate out-of-fold predictions
            for i, (name, model) in enumerate(selected_models.items()):
                self.logger.info(f"Generating meta-features for {name}...")
                test_preds = np.zeros(len(X_test))
                
                try:
                    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_fold_train = y_train_log.iloc[train_idx]
                        
                        model_clone = clone(model)
                        model_clone.fit(X_fold_train, y_fold_train)
                        meta_train[val_idx, i] = model_clone.predict(X_fold_val)
                        test_preds += model_clone.predict(X_test) / kf.n_splits
                        self.logger.debug(f"Completed fold {fold+1}/{kf.n_splits} for {name}")
                    
                    meta_test[:, i] = test_preds
                    model.fit(X_train, y_train_log)
                    self.logger.info(f"Generated meta-features for {name} successfully")
                except Exception as e:
                    self.logger.error(f"Error generating meta-features for {name}: {str(e)}")
                    # Fill with zeros if a model fails
                    meta_test[:, i] = 0.0
            
            # Try both a simple average and a meta-model
            self.logger.info("Training meta-model and computing simple average...")
            meta_model = Ridge(alpha=0.5, random_state=random_state)
            meta_model.fit(meta_train, y_train_log)
            
            # Meta-model predictions
            meta_preds = meta_model.predict(meta_test)
            
            # Simple average predictions
            avg_preds = np.mean(meta_test, axis=1)
            
            # Choose between meta-model and simple average based on CV performance
            meta_model_cv_score = 0
            avg_model_cv_score = 0
            
            for train_idx, val_idx in kf.split(meta_train):
                # Meta-model CV
                meta_model_clone = Ridge(alpha=0.5, random_state=random_state)
                meta_model_clone.fit(meta_train[train_idx], y_train_log.iloc[train_idx])
                meta_preds_cv = meta_model_clone.predict(meta_train[val_idx])
                meta_rmse = np.sqrt(mean_squared_error(y_train_log.iloc[val_idx], meta_preds_cv))
                meta_model_cv_score += meta_rmse / kf.n_splits
                
                # Average CV
                avg_preds_cv = np.mean(meta_train[val_idx], axis=1)
                avg_rmse = np.sqrt(mean_squared_error(y_train_log.iloc[val_idx], avg_preds_cv))
                avg_model_cv_score += avg_rmse / kf.n_splits
            
            self.logger.info(f"Meta-model CV RMSE: {meta_model_cv_score:.4f}")
            self.logger.info(f"Simple average CV RMSE: {avg_model_cv_score:.4f}")
            
            # Return the better of the two approaches
            if meta_model_cv_score <= avg_model_cv_score:
                self.logger.info("Using meta-model for final predictions")
                return meta_preds
            else:
                self.logger.info("Using simple average for final predictions")
                return avg_preds
        else:
            self.logger.error("No models evaluated successfully, returning zeros")
            return np.zeros(len(X_test))
        
    def handle_outliers(self, X_train, y_train):
        """Identify and handle outliers in training data"""
        self.logger.info("Analyzing outliers in target variable")
        
        from scipy import stats
        
        # Calculate z-scores for the target
        z_scores = np.abs(stats.zscore(y_train))
        
        # Identify potential outliers (z-score > 3)
        outlier_indices = np.where(z_scores > 3)[0]
        
        if len(outlier_indices) > 0:
            self.logger.info(f"Identified {len(outlier_indices)} potential outliers in target ({len(outlier_indices)/len(y_train)*100:.2f}%)")
            
            # Option 1: Remove outliers
            # X_train_clean = X_train.drop(outlier_indices)
            # y_train_clean = y_train.drop(outlier_indices)
            
            # Option 2: Cap outliers at a threshold (better for prediction)
            y_train_clean = y_train.copy()
            
            # Calculate IQR
            Q1 = y_train.quantile(0.25)
            Q3 = y_train.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count values outside bounds
            below_count = (y_train < lower_bound).sum()
            above_count = (y_train > upper_bound).sum()
            
            # Cap outliers
            y_train_clean.loc[y_train < lower_bound] = lower_bound
            y_train_clean.loc[y_train > upper_bound] = upper_bound
            
            self.logger.info(f"Applied capping to outliers using IQR method: {below_count} low outliers, {above_count} high outliers")
            self.logger.info(f"IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            return X_train, y_train_clean
    
        self.logger.info("No significant outliers detected in target variable")
        return X_train, y_train
    
    def try_different_transformations(self, y_train):
        """Try different transformations and select the best one"""
        self.logger.info("Evaluating different target variable transformations")
        
        from scipy import stats
        import numpy as np
        import warnings
        
        # Log transformation 
        y_log = np.log1p(y_train)
        
        # Box-Cox transformation (requires positive values)
        try:
            with warnings.catch_warnings():
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    # Fix: boxcox_normmax returns just the lambda value
                    lambda_param = stats.boxcox_normmax(y_train)
                    y_boxcox = stats.boxcox(y_train, lambda_param)
                    boxcox_details = {'name': 'boxcox', 
                                'lambda': lambda_param}
                    self.logger.debug(f"Box-Cox lambda: {lambda_param:.4f}")
        except Exception as e:
            self.logger.warning(f"Box-Cox transformation failed: {str(e)}")
            y_boxcox = y_log
            boxcox_details = None
        
        # Add Yeo-Johnson transformation (works with negative values)
        try:
            with warnings.catch_warnings():
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    yj_param = stats.yeojohnson_normmax(y_train)
                    y_yeojohnson = stats.yeojohnson(y_train, yj_param)
                    yeojohnson_details = {'name': 'yeojohnson', 
                                    'lambda': yj_param}
                    self.logger.debug(f"Yeo-Johnson lambda: {yj_param:.4f}")
        except Exception as e:
            self.logger.warning(f"Yeo-Johnson transformation failed: {str(e)}")
            y_yeojohnson = y_log
            yeojohnson_details = None
        
        # Square root transformation
        y_sqrt = np.sqrt(y_train)
        
        # Check normality of each transformation
        from scipy.stats import shapiro
        
        # Use a sample of 5000 points for large datasets
        if len(y_train) > 5000:
            self.logger.debug("Using 5000 sample points for normality tests due to large dataset")
            idx = np.random.choice(len(y_train), 5000, replace=False)
            log_stat, log_p = shapiro(y_log.iloc[idx])
            boxcox_stat, boxcox_p = shapiro(y_boxcox[idx]) if boxcox_details else (0, 0)
            yeojohnson_stat, yeojohnson_p = shapiro(y_yeojohnson[idx]) if yeojohnson_details else (0, 0)
            sqrt_stat, sqrt_p = shapiro(y_sqrt.iloc[idx])
        else:
            log_stat, log_p = shapiro(y_log)
            boxcox_stat, boxcox_p = shapiro(y_boxcox) if boxcox_details else (0, 0)
            yeojohnson_stat, yeojohnson_p = shapiro(y_yeojohnson) if yeojohnson_details else (0, 0)
            sqrt_stat, sqrt_p = shapiro(y_sqrt)
        
        # Higher p-value indicates more normality
        p_values = {
            'log': log_p, 
            'boxcox': boxcox_p if boxcox_details else 0, 
            'yeojohnson': yeojohnson_p if yeojohnson_details else 0,
            'sqrt': sqrt_p
        }
        best_transform = max(p_values, key=p_values.get)
        
        self.logger.info(f"Transformation normality tests (p-values, higher is better):")
        self.logger.info(f"Log: {log_p:.6f}")
        self.logger.info(f"Box-Cox: {boxcox_p:.6f}" if boxcox_details else "Box-Cox: Not available")
        self.logger.info(f"Yeo-Johnson: {yeojohnson_p:.6f}" if yeojohnson_details else "Yeo-Johnson: Not available")
        self.logger.info(f"Square root: {sqrt_p:.6f}")
        self.logger.info(f"Selected: {best_transform} transformation")
        
        # Return the best transformation and details
        if best_transform == 'log':
            return y_log, {'name': 'log', 'transform': np.log1p, 'inverse': np.expm1}
        elif best_transform == 'boxcox' and boxcox_details:
            # Important: Convert the numpy array to a pandas Series with the same index
            return pd.Series(y_boxcox, index=y_train.index), {
                'name': 'boxcox', 
                'lambda': lambda_param,
                'inverse': lambda x: special.inv_boxcox(x, lambda_param)
            }
        elif best_transform == 'yeojohnson' and yeojohnson_details:
            # Important: Convert the numpy array to a pandas Series with the same index
            return pd.Series(y_yeojohnson, index=y_train.index), {
                'name': 'yeojohnson', 
                'lambda': yj_param,
                'inverse': lambda x: stats.inv_yeojohnson(x, yj_param)
            }
        else:
            return y_sqrt, {'name': 'sqrt', 'transform': np.sqrt, 'inverse': np.square}

if __name__ == "__main__":
    pipeline = RefinedPipeline()
    submission = pipeline.run()