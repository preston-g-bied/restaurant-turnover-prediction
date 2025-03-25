# src/models/advanced_pipeline.py

import pandas as pd
import numpy as np
import os
import logging
import time
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Lasso, ElasticNet, HuberRegressor, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'logs', 'advanced_pipeline.log'))
    ]
)
logger = logging.getLogger()

class AdvancedRegressionPipeline:
    """
    Advanced regression pipeline with improved feature engineering, model ensembling,
    and robust evaluation for restaurant turnover prediction.
    """

    def __init__(self, random_state=42):
        """
        Initialize the pipeline.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.data_dir = Path(os.path.join(project_root, 'data', 'processed'))
        self.output_dir = Path(os.path.join(project_root, 'models'))
        self.submission_dir = Path(os.path.join(project_root, 'data', 'submissions'))
        self.random_state = random_state

        # create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.submission_dir.mkdir(exist_ok=True, parents=True)

        # initialize containers
        self.base_models = {}
        self.meta_models = {}
        self.feature_importances = {}
        self.model_errors = {}
        self.predictions = {}

        # track best model
        self.best_model = None
        self.best_score = float('inf')
        self.best_model_name = None

        # scalers
        self.feature_scaler = None
        self.target_transformer = TargetTransformer()

    def load_data(self):
        """
        Load the processed training and test data.
        
        Returns:
        --------
        X_train, y_train, X_test, test_ids : DataFrames and Series
            Training and test data, target variable and test IDs
        """
        logger.info('Loading data...')

        # load training data
        train_data = pd.read_csv(self.data_dir / 'train_processed.csv')

        # extract target and features
        y_train = train_data['Annual Turnover']
        X_train = train_data.drop('Annual Turnover', axis=1)

        # load test data
        test_data = pd.read_csv(self.data_dir / 'test_processed.csv')
        test_ids = test_data['Registration Number']
        X_test = test_data.drop('Registration Number', axis=1)

        # ensure X_train and X_test have the same columns
        common_cols = list(set(X_train.columns).intersection(X_test.columns))
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]

        logger.info(f"Loaded {X_train.shape[0]} training samples with {X_train.shape[1]} features")
        logger.info(f"Loaded {X_test.shape[0]} test samples")

        return X_train, y_train, X_test, test_ids
    
    def enhance_features(self, X_train, y_train, X_test):
        """
        Apply advanced feature engineering techniques.
        
        Parameters:
        -----------
        X_train, X_test : DataFrames
            Training and test features
        y_train : Series
            Target variable
            
        Returns:
        --------
        X_train_enhanced, X_test_enhanced : DataFrames
            Enhanced feature sets
        """
        logger.info('Enhancing features...')

        # create copies to avoid modifying originals
        X_train_enhanced = X_train.copy()
        X_test_enhanced = X_test.copy()

        # 1. handle outliers in numeric features
        numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns

        # apply robust scaling to reduce impact of outliers
        self.feature_scaler = RobustScaler()
        X_train_enhanced[numeric_cols] = self.feature_scaler.fit_transform(X_train[numeric_cols])
        X_test_enhanced[numeric_cols] = self.feature_scaler.transform(X_test[numeric_cols])

        # 2. create additional interaction features for high-importance features
        # identify important features based on correlation with target
        corr_with_target = []
        for col in X_train.columns:
            try:
                corr = X_train[col].corr(y_train)
                corr_with_target.append((col, abs(corr)))
            except:
                pass

        # sort by absolute correlation
        corr_with_target.sort(key=lambda x: x[1], reverse=True)
        top_features = [item[0] for item in corr_with_target[:10]]  # top 10 features

        # create interactions between top features
        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i+1:]:
                # Multiplicative interaction
                X_train_enhanced[f'{feat1}_{feat2}_mult'] = X_train[feat1] * X_train[feat2]
                X_test_enhanced[f'{feat1}_{feat2}_mult'] = X_test[feat1] * X_test[feat2]
                    
                # Ratio interaction (handle division by zero)
                X_train_enhanced[f'{feat1}_{feat2}_ratio'] = X_train[feat1] / (X_train[feat2] + 1e-8)
                X_test_enhanced[f'{feat1}_{feat2}_ratio'] = X_test[feat1] / (X_test[feat2] + 1e-8)
                    
                # Sum interaction
                X_train_enhanced[f'{feat1}_{feat2}_sum'] = X_train[feat1] + X_train[feat2]
                X_test_enhanced[f'{feat1}_{feat2}_sum'] = X_test[feat1] + X_test[feat2]

        # 3. create polynomial features for top features
        for feat in top_features[:5]:
            if feat in X_train.columns:
                # Squared terms
                X_train_enhanced[f'{feat}_squared'] = X_train[feat] ** 2
                X_test_enhanced[f'{feat}_squared'] = X_test[feat] ** 2
                
                # Cubic terms
                X_train_enhanced[f'{feat}_cubed'] = X_train[feat] ** 3
                X_test_enhanced[f'{feat}_cubed'] = X_test[feat] ** 3

        # 4. create domain-specific composite features

        # Restaurant quality score (if relevant features exist)
        quality_features = ['Food Rating', 'Hygiene Rating', 'Overall Restaurant Rating']
        existing_quality_features = [f for f in quality_features if f in X_train.columns]
        
        if existing_quality_features:
            X_train_enhanced['quality_composite'] = X_train[existing_quality_features].mean(axis=1)
            X_test_enhanced['quality_composite'] = X_test[existing_quality_features].mean(axis=1)
        
        # Social media impact score
        social_features = ['Facebook Popularity Quotient', 'Instagram Popularity Quotient']
        existing_social_features = [f for f in social_features if f in X_train.columns]
        
        if existing_social_features:
            X_train_enhanced['social_composite'] = X_train[existing_social_features].mean(axis=1)
            X_test_enhanced['social_composite'] = X_test[existing_social_features].mean(axis=1)
        
        # Entertainment value score
        entertainment_features = ['Live Music Rating', 'Comedy Gigs Rating', 
                                 'Value Deals Rating', 'Live Sports Rating']
        existing_ent_features = [f for f in entertainment_features if f in X_train.columns]

        if existing_ent_features:
            # First handle missing values
            train_ent = X_train[existing_ent_features].fillna(0)
            test_ent = X_test[existing_ent_features].fillna(0)
            
            # Create composite
            X_train_enhanced['entertainment_composite'] = train_ent.mean(axis=1)
            X_test_enhanced['entertainment_composite'] = test_ent.mean(axis=1)
            
            # Also create "has entertainment" feature
            X_train_enhanced['has_entertainment'] = (train_ent.sum(axis=1) > 0).astype(int)
            X_test_enhanced['has_entertainment'] = (test_ent.sum(axis=1) > 0).astype(int)
        
        logger.info(f"Enhanced feature set: {X_train_enhanced.shape[1]} features")
        
        return X_train_enhanced, X_test_enhanced
    
    def setup_models(self):
        """
        Set up base models with carefully chosen hyperparameters.
        """
        logger.info("Setting up models...")
        
        # Linear models with different regularization approaches
        self.base_models['lasso'] = Lasso(
            alpha=0.001,  # Start with a smaller alpha than default
            max_iter=5000,
            random_state=self.random_state
        )
        
        self.base_models['elastic_net'] = ElasticNet(
            alpha=0.001,
            l1_ratio=0.7,  # More L1 than L2 regularization
            max_iter=5000,
            random_state=self.random_state
        )
        
        self.base_models['ridge'] = Ridge(
            alpha=1.0,
            random_state=self.random_state
        )
        
        self.base_models['huber'] = HuberRegressor(
            epsilon=1.35,  # Default value
            max_iter=500,
            alpha=0.0001
        )

        # Tree-based models
        self.base_models['gb'] = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=self.random_state
        )
        
        self.base_models['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state
        )
        
        # Advanced boosting libraries
        self.base_models['lgb'] = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state
        )

        self.base_models['xgb'] = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state
        )
        
        self.base_models['cat'] = cb.CatBoostRegressor(
            iterations=300,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            random_strength=1,
            random_seed=self.random_state,
            verbose=0
        )
        
        # Meta-models for stacking
        self.meta_models['elastic_net'] = ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            max_iter=5000,
            random_state=self.random_state
        )

        self.meta_models['ridge'] = Ridge(
            alpha=1.0,
            random_state=self.random_state
        )
        
        self.meta_models['gb'] = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            random_state=self.random_state
        )
        
        logger.info(f"Set up {len(self.base_models)} base models and {len(self.meta_models)} meta-models")

    def train_and_evaluate_base_models(self, X_train, y_train, cv=5):
        """
        Train and evaluate all base models with cross-validation.
        
        Parameters:
        -----------
        X_train : DataFrame
            Training features
        y_train : Series
            Target variable
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        mean_scores : dict
            Dictionary of mean scores for each model
        """
        logger.info(f"Training and evaluating base models with {cv}-fold CV...")
        
        # Initialize containers for results
        cv_scores = {}
        cv_predictions = {}
        mean_scores = {}
        
        # Create cross-validation splitter
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Iterate through models
        for name, model in self.base_models.items():
            logger.info(f"Training {name}...")
            start_time = time.time()
            
            # Initialize arrays for scores and predictions
            fold_scores = []
            fold_predictions = np.zeros(len(X_train))
            
            # Perform cross-validation
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                # Split data
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Train model
                model.fit(X_fold_train, y_fold_train)
                
                # Make predictions
                y_pred = model.predict(X_fold_val)
                
                # Store predictions in the right places
                fold_predictions[val_idx] = y_pred
                
                # Calculate and store score
                rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
                r2 = r2_score(y_fold_val, y_pred)
                fold_scores.append(rmse)
                
                logger.info(f"{name} - Fold {fold+1}: RMSE = {rmse:.2f}, R² = {r2:.4f}")

            # Calculate mean score
            mean_rmse = np.mean(fold_scores)
            std_rmse = np.std(fold_scores)
            
            # Store results
            cv_scores[name] = fold_scores
            cv_predictions[name] = fold_predictions
            mean_scores[name] = mean_rmse
            
            # Training time
            train_time = time.time() - start_time
            
            logger.info(f"{name} - Mean RMSE: {mean_rmse:.2f} ±{std_rmse:.2f} (time: {train_time:.2f}s)")
            
            # Check if this is the best model
            if mean_rmse < self.best_score:
                self.best_score = mean_rmse
                self.best_model_name = name
        
        # Store predictions for later analysis and stacking
        self.predictions = cv_predictions

        logger.info(f"Best base model: {self.best_model_name} with RMSE: {self.best_score:.2f}")
        
        return mean_scores
    
    def stack_models(self, X_train, y_train, X_test, cv=5):
        """
        Create a stacked model using predictions from base models.
        
        Parameters:
        -----------
        X_train : DataFrame
            Training features
        y_train : Series
            Target variable
        X_test : DataFrame
            Test features
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        test_preds : ndarray
            Predictions for test set
        """
        logger.info("Creating stacked model...")
        
        # Get list of base models to include (can be all or a subset)
        base_models_to_stack = list(self.base_models.keys())
        
        # Check if we already have CV predictions from base models
        if not self.predictions:
            raise ValueError("No base model predictions found. Run train_and_evaluate_base_models first.")
        
        # Create a matrix of meta-features for training
        meta_features = np.column_stack([self.predictions[name] for name in base_models_to_stack])
        meta_features_df = pd.DataFrame(
            meta_features, 
            columns=[f"{name}_pred" for name in base_models_to_stack]
        )
        
        # Get predictions on full test set
        test_meta_features = []

        # Train each base model on full training data and predict test set
        for name in base_models_to_stack:
            logger.info(f"Getting test predictions from {name}...")
            model = self.base_models[name]
            model.fit(X_train, y_train)
            test_preds = model.predict(X_test)
            test_meta_features.append(test_preds)
        
        # Stack test predictions
        test_meta_features = np.column_stack(test_meta_features)
        test_meta_features_df = pd.DataFrame(
            test_meta_features,
            columns=[f"{name}_pred" for name in base_models_to_stack]
        )
        
        # Try each meta-model on the stacked features
        meta_cv_scores = {}

        for meta_name, meta_model in self.meta_models.items():
            logger.info(f"Evaluating meta-model: {meta_name}...")
            
            # Evaluate with cross-validation
            kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(meta_features)):
                # Split meta-features
                meta_train, meta_val = meta_features[train_idx], meta_features[val_idx]
                y_meta_train, y_meta_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Train meta-model
                meta_model.fit(meta_train, y_meta_train)
                
                # Evaluate
                meta_preds = meta_model.predict(meta_val)
                rmse = np.sqrt(mean_squared_error(y_meta_val, meta_preds))
                fold_scores.append(rmse)
            
            # Calculate mean score
            mean_rmse = np.mean(fold_scores)
            meta_cv_scores[meta_name] = mean_rmse
            
            logger.info(f"Meta-model {meta_name} - Mean RMSE: {mean_rmse:.2f}")
        
        # Select best meta-model
        best_meta_name = min(meta_cv_scores, key=meta_cv_scores.get)
        best_meta_score = meta_cv_scores[best_meta_name]
        
        logger.info(f"Best meta-model: {best_meta_name} with RMSE: {best_meta_score:.2f}")
        
        # Train best meta-model on all stacked features
        best_meta_model = self.meta_models[best_meta_name]
        best_meta_model.fit(meta_features, y_train)
        
        # Make final predictions on test set
        test_preds = best_meta_model.predict(test_meta_features)

        # Store best model
        if best_meta_score < self.best_score:
            self.best_score = best_meta_score
            self.best_model_name = f"stacked_{best_meta_name}"
            
            # Create and store the full stacked model for later use
            self.best_model = {
                'base_models': {name: self.base_models[name] for name in base_models_to_stack},
                'meta_model': best_meta_model,
                'meta_model_name': best_meta_name
            }
        
        return test_preds
    
    def blend_models(self, X_train, y_train, X_test, weights=None):
        """
        Create a simple weighted ensemble of models.
        
        Parameters:
        -----------
        X_train : DataFrame
            Training features
        y_train : Series
            Target variable
        X_test : DataFrame
            Test features
        weights : dict, optional
            Dictionary of model weights. If None, use inverse of CV scores as weights.
            
        Returns:
        --------
        test_preds : ndarray
            Predictions for test set
        """
        logger.info("Creating blended model ensemble...")
        
        # Train all base models on full training data
        all_preds = {}
        
        for name, model in self.base_models.items():
            logger.info(f"Training {name} for blending...")
            model.fit(X_train, y_train)
            test_preds = model.predict(X_test)
            all_preds[name] = test_preds
        
        # Calculate weights if not provided
        if weights is None:
            # Use inverse of CV scores as weights
            cv_scores = {name: np.mean(scores) for name, scores in self.model_errors.items()}
            
            # Convert to weights (lower error = higher weight)
            sum_errors = sum(1/score for score in cv_scores.values())
            weights = {name: (1/score)/sum_errors for name, score in cv_scores.items()}
            
            logger.info(f"Calculated weights: {weights}")
        
        # Create weighted average prediction
        weighted_preds = np.zeros(len(X_test))
        
        for name, preds in all_preds.items():
            if name in weights:
                weighted_preds += weights[name] * preds
        
        return weighted_preds
    
    def optimize_ensemble_weights(self, X_train, y_train, X_test, n_trials=100):
        """
        Optimize weights for ensemble using random search.
        
        Parameters:
        -----------
        X_train : DataFrame
            Training features
        y_train : Series
            Target variable
        X_test : DataFrame
            Test features
        n_trials : int
            Number of random trials for weight optimization
            
        Returns:
        --------
        test_preds : ndarray
            Predictions for test set
        """
        logger.info(f"Optimizing ensemble weights with {n_trials} trials...")

        # Split data for optimization
        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state
        )
        
        # Train all models on the same training set
        val_preds = {}
        test_preds = {}
        
        for name, model in self.base_models.items():
            # Train on training portion
            model.fit(X_train_main, y_train_main)
            
            # Predict on validation set
            val_preds[name] = model.predict(X_val)
            
            # Predict on test set
            test_preds[name] = model.predict(X_test)
        
        # Random search for optimal weights
        best_rmse = float('inf')
        best_weights = None

        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        
        for trial in range(n_trials):
            # Generate random weights
            raw_weights = np.random.rand(len(self.base_models))
            
            # Normalize weights to sum to 1
            weights = raw_weights / raw_weights.sum()
            
            # Create weighted prediction
            weighted_val_pred = np.zeros(len(y_val))
            
            for i, name in enumerate(self.base_models.keys()):
                weighted_val_pred += weights[i] * val_preds[name]
            
            # Evaluate
            rmse = np.sqrt(mean_squared_error(y_val, weighted_val_pred))
            
            # Update best weights if improvement
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = weights
                logger.info(f"New best ensemble - Trial {trial+1}: RMSE = {rmse:.2f}")
        
        # Create dictionary of weights
        weight_dict = {name: weight for name, weight in zip(self.base_models.keys(), best_weights)}
        
        # Create final weighted prediction for test set
        weighted_test_pred = np.zeros(len(X_test))
        
        for i, name in enumerate(self.base_models.keys()):
            weighted_test_pred += best_weights[i] * test_preds[name]
        
        logger.info(f"Optimized ensemble - Final weights: {weight_dict}")
        logger.info(f"Optimized ensemble - Validation RMSE: {best_rmse:.2f}")
        
        # Save the weights for later reference
        self.ensemble_weights = weight_dict
        
        # Check if this is our new best model
        if best_rmse < self.best_score:
            self.best_score = best_rmse
            self.best_model_name = "optimized_ensemble"
            self.best_model = {
                'type': 'weighted_ensemble',
                'weights': weight_dict,
                'models': self.base_models
            }
        
        return weighted_test_pred
    
    def segment_based_ensemble(self, X_train, y_train, X_test, segment_col=None, n_segments=3):
        """
        Create an ensemble where different models are used for different segments of data.
        
        Parameters:
        -----------
        X_train : DataFrame
            Training features
        y_train : Series
            Target variable
        X_test : DataFrame
            Test features
        segment_col : str, optional
            Column to use for segmentation. If None, use target value.
        n_segments : int
            Number of segments to create if using target value
            
        Returns:
        --------
        test_preds : ndarray
            Predictions for test set
        """
        logger.info("Creating segment-based ensemble...")

        # Create segments
        if segment_col is not None and segment_col in X_train.columns:
            # Use provided column for segmentation
            segments = X_train[segment_col].values
            logger.info(f"Using {segment_col} for segmentation")
            
            # Determine how to segment test data
            test_segments = X_test[segment_col].values

        else:
            # Use target variable segments
            # For test data, we'll predict segment first
            logger.info(f"Using target variable quantiles for segmentation")
            
            # Create segment boundaries using quantiles
            segment_edges = np.quantile(y_train, np.linspace(0, 1, n_segments+1))
            
            # Assign each training point to a segment
            segments = np.zeros(len(y_train), dtype=int)
            for i in range(1, len(segment_edges)):
                segments[(y_train >= segment_edges[i-1]) & (y_train < segment_edges[i])] = i-1
            
            # Train a classifier to predict segments for test data
            from sklearn.ensemble import RandomForestClassifier
            segment_clf = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state
            )
            segment_clf.fit(X_train, segments)
            
            # Predict segments for test data
            test_segments = segment_clf.predict(X_test)
        
        # Find unique segments
        unique_segments = np.unique(segments)

        # Train models for each segment
        segment_models = {}
        segment_rmse = {}

        for segment in unique_segments:
            logger.info(f"Training models for segment {segment}...")
            
            # Get data for this segment
            segment_mask = segments == segment
            X_segment = X_train[segment_mask]
            y_segment = y_train[segment_mask]
            
            # Train models for this segment
            segment_model_scores = {}
            
            for name, model in self.base_models.items():
                # Split for validation
                X_seg_train, X_seg_val, y_seg_train, y_seg_val = train_test_split(
                    X_segment, y_segment, test_size=0.2, random_state=self.random_state
                )
                
                # Train model
                model.fit(X_seg_train, y_seg_train)
                
                # Evaluate
                y_seg_pred = model.predict(X_seg_val)
                rmse = np.sqrt(mean_squared_error(y_seg_val, y_seg_pred))
                
                segment_model_scores[name] = rmse
            
            # Select best model for this segment
            best_model_name = min(segment_model_scores, key=segment_model_scores.get)
            best_model = self.base_models[best_model_name]
            best_rmse = segment_model_scores[best_model_name]
            
            # Train on all segment data
            best_model.fit(X_segment, y_segment)

            # Store
            segment_models[segment] = {
                'model': best_model,
                'model_name': best_model_name
            }
            segment_rmse[segment] = best_rmse
            
            logger.info(f"Segment {segment} - Best model: {best_model_name}, RMSE: {best_rmse:.2f}")
        
        # Make predictions on test data
        test_preds = np.zeros(len(X_test))
        
        for segment in unique_segments:
            # Find test samples in this segment
            segment_mask = test_segments == segment
            
            if np.sum(segment_mask) > 0:
                # Get model for this segment
                model = segment_models[segment]['model']
                
                # Make predictions
                segment_preds = model.predict(X_test[segment_mask])
                
                # Store
                test_preds[segment_mask] = segment_preds

        # Store segment models
        self.segment_models = segment_models
        
        # Calculate weighted average RMSE across segments
        segment_sizes = {segment: np.sum(segments == segment) for segment in unique_segments}
        total_samples = sum(segment_sizes.values())
        weighted_rmse = sum(segment_rmse[segment] * segment_sizes[segment] / total_samples 
                           for segment in unique_segments)
        
        logger.info(f"Segment-based ensemble - Weighted RMSE: {weighted_rmse:.2f}")
        
        # Check if this is our new best model
        if weighted_rmse < self.best_score:
            self.best_score = weighted_rmse
            self.best_model_name = "segment_based_ensemble"
            self.best_model = {
                'type': 'segment_based',
                'segment_models': segment_models,
                'segment_clf': segment_clf if 'segment_clf' in locals() else None,
                'segment_edges': segment_edges if 'segment_edges' in locals() else None
            }
        
        return test_preds
    
    def run_pipeline(self, apply_target_transform=True):
        """
        Run the full modeling pipeline.
        
        Parameters:
        -----------
        apply_target_transform : bool
            Whether to apply log transformation to the target variable
            
        Returns:
        --------
        submission : DataFrame
            Final submission file with predictions
        """
        logger.info('Running full pipeline...')

        # 1. load data
        X_train, y_train, X_test, test_ids = self.load_data()

        # 2. apply target transformation if requested
        if apply_target_transform:
            y_train_original = y_train.copy()
            y_train, transform_details = self.target_transformer.fit_transform(y_train)
            logger.info(f"Applied {transform_details['name']} transformation to target")

        # 3. enhance features
        X_train_enhanced, X_test_enhanced = self.enhance_features(X_train, y_train, X_test)

        # 4. setup models
        self.setup_models()

        # 5. train and evaluate base models
        base_scores = self.train_and_evaluate_base_models(X_train_enhanced, y_train)

        # save information about the best base model
        best_base_model_name = min(base_scores, key=base_scores.get)
        best_base_model_score = base_scores[best_base_model_name]

        # 6. create stacked ensemble
        stacked_preds = self.stack_models(X_train_enhanced, y_train, X_test_enhanced)

        # 7. optimize ensemble weights
        weighted_preds = self.optimize_ensemble_weights(X_train_enhanced, y_train, X_test_enhanced)

        # 8. create segment-based ensemble
        segment_preds = self.segment_based_ensemble(X_train_enhanced, y_train, X_test_enhanced)

        # 9. prepare final predictions based on best model
        logger.info(f"Best model: {self.best_model_name} with score: {self.best_score:.2f}")
        
        if self.best_model_name == "optimized_ensemble":
            final_preds = weighted_preds
        elif self.best_model_name == "segment_based_ensemble":
            final_preds = segment_preds
        elif self.best_model_name.startswith("stacked_"):
            final_preds = stacked_preds
        else:
            # Best model was a base model
            best_model = self.base_models[self.best_model_name]
            best_model.fit(X_train_enhanced, y_train)
            final_preds = best_model.predict(X_test_enhanced)
        
        # 10. Inverse transform predictions if needed
        if apply_target_transform:
            final_preds = self.target_transformer.inverse_transform(final_preds)
        
        # 11. Create submission file
        submission = pd.DataFrame({
            'Registration Number': test_ids,
            'Annual Turnover': final_preds
        })
        
        # 12. Save final model and submission
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save model
        model_path = self.output_dir / f"{self.best_model_name}_{timestamp}.joblib"
        joblib.dump(self.best_model, model_path)
        
        # Save submission
        submission_path = self.submission_dir / f"submission_{self.best_model_name}_{timestamp}.csv"
        submission.to_csv(submission_path, index=False)
        
        logger.info(f"Saved best model to {model_path}")
        logger.info(f"Saved submission to {submission_path}")
        logger.info(f"Submission statistics - Min: {final_preds.min():.2f}, "
                   f"Max: {final_preds.max():.2f}, Mean: {final_preds.mean():.2f}")
        
        return submission

class TargetTransformer:
    """
    A class for transforming and inverse transforming the target variable.
    """

    def __init__(self):
        """Initialize the transformer."""
        self.transformer = None
        self.inverse_transformer = None
        self.transform_details = None

    def fit_transform(self, y):
        """
        Find the best transformation for the target variable.
        
        Parameters:
        -----------
        y : Series
            Target variable
            
        Returns:
        --------
        y_transformed : Series
            Transformed target
        """
        from scipy import stats
        import numpy as np

        # store original data
        self.original_target = y.copy()

        # try different transformations
        transformations = {}

        # log transformations
        transformations['log'] = {
            'transform': lambda x: np.log1p(x),
            'inverse': lambda x: np.expm1(x),
            'name': 'log'
        }

        # square root transformation
        transformations['sqrt'] = {
            'transform': lambda x: np.sqrt(x),
            'inverse': lambda x: np.square(x),
            'name': 'sqrt'
        }

        # box-cox transformation (requires positive values)
        if y.min() > 0:
            try:
                boxcox_lambda = stats.boxcox_normmax(y)
                transformations['boxcox'] = {
                    'transform': lambda x: stats.boxcox(x, boxcox_lambda),
                    'inverse': lambda x: stats.inv_boxcox(x, boxcox_lambda),
                    'lambda': boxcox_lambda,
                    'name': 'boxcox'
                }
            except:
                pass

        # yeo-johnson transformation
        try:
            yeojohnson_lambda = stats.yeojohnson_normax(y)

            # define inverse yeojohnson_lambda
            def inverse_yeojohnson(x, lmbda):
                """
                Inverse of the Yeo-Johnson transformation.
                
                Parameters:
                -----------
                x : array-like
                    Transformed data
                lmbda : float
                    Transformation parameter
                    
                Returns:
                --------
                x_inv : array-like
                    Original scale data
                """
                x_inv = np.zeros_like(x, dtype=float)
        
                # x >= 0 and lambda != 0
                pos_nonzero = (x >= 0) & (np.abs(lmbda) > 1e-8)
                if np.any(pos_nonzero):
                    x_inv[pos_nonzero] = np.power(x[pos_nonzero] * lmbda + 1, 1/lmbda) - 1
                
                # x >= 0 and lambda == 0
                pos_zero = (x >= 0) & (np.abs(lmbda) <= 1e-8)
                if np.any(pos_zero):
                    x_inv[pos_zero] = np.exp(x[pos_zero]) - 1
                
                # x < 0 and lambda != 2
                neg_nonzero = (x < 0) & (np.abs(lmbda - 2) > 1e-8)
                if np.any(neg_nonzero):
                    x_inv[neg_nonzero] = 1 - np.power(-(2 - lmbda) * x[neg_nonzero] + 1, 1/(2 - lmbda))
                
                # x < 0 and lambda == 2
                neg_zero = (x < 0) & (np.abs(lmbda - 2) <= 1e-8)
                if np.any(neg_zero):
                    x_inv[neg_zero] = 1 - np.exp(-x[neg_zero])
                
                return x_inv

            transformations['yeojohnson'] = {
                'transform': lambda x: stats.yeojohnson(x, yeojohnson_lambda),
                'inverse': lambda x: inverse_yeojohnson(x, yeojohnson_lambda),
                'lambda': yeojohnson_lambda,
                'name': 'yeojohnson'
            }

        except:
            pass

        # evaluate each transformation for normality
        normality_scores = {}

        for name, transform_info in transformations.items():
            try:
                y_transformed = transform_info['transform'](y)

                # check normality using shapiro-wilk test
                # higher p-value indicates more normal distribution
                if len(y) > 5000:
                    # for large datasets, use a sample
                    sample_idx = np.random.choice(len(y), 5000, replace=False)
                    shapiro_stat, shapiro_p = stats.shapiro(y_transformed[sample_idx])
                else:
                    shapiro_stat, shapiro_p = stats.shapiro(y_transformed)

                normality_scores[name] = shapiro_p
            except:
                continue

        # select best transformation based on normality
        best_transform = max(normality_scores, key=normality_scores.get)

        # apply the best transformation
        self.transformer = transformations[best_transform]['transform']
        self.inverse_transformer = transformations[best_transform]['inverse']
        self.transform_details = transformations[best_transform]

        # transform and return
        y_transformed = self.transformer(y)

        return y_transformed, self.transform_details
    
    def transform(self, y):
        """
        Apply the transformation to new data.
        
        Parameters:
        -----------
        y : Series or ndarray
            Target variable
            
        Returns:
        --------
        y_transformed : ndarray
            Transformed target
        """
        if self.transformer is None:
            raise ValueError("Transformer not fitted. Call fit_transform first.")
        
        return self.transformer(y)
    
    def inverse_transform(self, y_transformed):
        """
        Apply the inverse transformation.
        
        Parameters:
        -----------
        y_transformed : Series or ndarray
            Transformed target variable
            
        Returns:
        --------
        y : ndarray
            Original-scale target
        """
        if self.inverse_transformer is None:
            raise ValueError("Transformer not fitted. Call fit_transform first.")
        
        return self.inverse_transformer(y_transformed)
    
if __name__ == "__main__":
    pipeline = AdvancedRegressionPipeline()
    submission = pipeline.run_pipeline(apply_target_transform=True)
    print(f"Pipeline completed with best model: {pipeline.best_model_name}")
    print(f"Best score: {pipeline.best_score:.2f}")