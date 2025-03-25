# src/models/ensemble.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger('ensemble')

class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Stacking ensemble that combines multiple base models using a meta-learner.
    """

    def __init__(self, base_models, meta_model, n_folds=5, use_features_in_meta=False, random_state=42):
        """
        Initialize the stacking ensemble.
        
        Parameters:
        -----------
        base_models : list of tuples
            List of (name, model) tuples for base models
        meta_model : estimator
            Model to use as meta-learner
        n_folds : int
            Number of folds for generating meta-features
        use_features_in_meta : bool
            Whether to use original features along with base model predictions in meta-model
        random_state : int
            Random state for reproducibility
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.use_features_in_meta = use_features_in_meta
        self.random_state = random_state
        self.base_models_fitted = []
        self.feature_importances_ = None

    def fit(self, X, y):
        """
        Fit the stacking ensemble.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Training features
        y : pandas.Series
            Target variable
        
        Returns:
        --------
        self
        """
        self.base_models_fitted = []

        # generate out-of-fold predictions for each base model
        meta_features = np.zeros((X.shape[0], len(self.base_models)))

        # define cross-validation strategy
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # for each base model
        for i, (name, model) in enumerate(self.base_models):
            logger.info(f'Generating meta-features for {name}')

            # for each fold
            for train_idx, val_idx in kf.split(X):
                # split data
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train = y.iloc[train_idx]

                # fit model on training fold
                fold_model = model.fit(X_fold_train, y_fold_train)

                # generate predictions for validation fold
                meta_features[val_idx, i] = fold_model.predict(X_fold_val)

            # fit model on full dataset for future predictions
            self.base_models_fitted.append((name, model.fit(X, y)))

        # prepare meta-features for meta-model
        if self.use_features_in_meta:
            # concatenate base model predictions with original features
            meta_features_df = pd.DataFrame(meta_features, columns=[f'model_{i}' for i in range(len(self.base_models))])
            X_meta = pd.concat([meta_features_df, X.reset_index(drop=True)], axis=1)
        else:
            # use only base model predictions
            X_meta = pd.DataFrame(meta_features, columns=[f'model_{i}' for i in range(len(self.base_models))])

        # fit meta-model
        logger.info('Fitting meta-model')
        self.meta_model.fit(X_meta, y)

        # try to extract feature importances if available
        if hasattr(self.meta_model, 'feature_importances_'):
            self.feature_importances_ = {
                f'meta_{col}': importance for col, importance in
                zip(X_meta.columns, self.meta_model.feature_importances_)
            }

            # add base model feature importances if available
            for i, (name, model) in enumerate(self.base_models_fitted):
                if hasattr(model, 'feature_importances_'):
                    weight = self.meta_model.feature_importances_[i] if i < len(self.meta_model.feature_importances_) else 1.0
                    for j, feat in enumerate(X.columns):
                        key = f'{name}_{feat}'
                        self.feature_importances_[key] = model.feature_importances_[j] * weight

        return self
    
    def predict(self, X):
        """
        Generate predictions for test data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Test features
        
        Returns:
        --------
        numpy.ndarray
            Predictions
        """
        # generate predictions from base models
        meta_features = np.column_stack([
            model.predict(X) for _, model in self.base_models_fitted
        ])

        # prepare meta-features for meta-model
        if self.use_features_in_meta:
            # concatenate base model predictions with original features
            meta_features_df = pd.DataFrame(meta_features, columns=[f'model_{i}' for i in range(len(self.base_models))])
            X_meta = pd.concat([meta_features_df, X.reset_index(drop=True)], axis=1)
        else:
            # use only base model predictions
            X_meta = pd.DataFrame(meta_features, columns=[f'model_{i}' for i in range(len(self.base_models))])

        # generate predictions from meta-model
        return self.meta_model.predict(X_meta)
    
class VotingEnsemble(BaseEstimator, RegressorMixin):
    """
    Voting ensemble that combines predictions through weighted averaging.
    """

    def __init__(self, models, weights=None):
        """
        Initialize the voting ensemble.
        
        Parameters:
        -----------
        models : list of tuples
            List of (name, model) tuples
        weights : list of float, optional
            Weights for each model (default: equal weights)
        """
        self.models = models
        self.weights = weights
        self.models_fitted = []

    def fit(self, X, y):
        """
        Fit the voting ensemble.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Training features
        y : pandas.Series
            Target variable
        
        Returns:
        --------
        self
        """
        self.models_fitted = []
        
        # fit each model
        for name, model in self.models:
            logger.info(f'Fitting {name}')
            self.models_fitted.append((name, model.fit(X, y)))

        # if weights are not provided use equal weights
        if self.weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)

        # normalize weights
        self.weights = np.array(self.weights) / sum(self.weights)

        return self
    
    def predict(self, X):
        """
        Generate predictions for test data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Test features
        
        Returns:
        --------
        numpy.ndarray
            Predictions
        """
        # generate predictions from each model
        predictions = np.column_stack([
            model.predict(X) for _, model in self.models_fitted
        ])

        # weighted average
        return predictions @ self.weights
    
class BlendingEnsemble(BaseEstimator, RegressorMixin):
    """
    Blending ensemble that uses a validation set to train the meta-model.
    """

    def __init__(self, base_models, meta_model, val_size=0.2, random_state=42, use_features_in_meta=False):
        """
        Initialize the blending ensemble.
        
        Parameters:
        -----------
        base_models : list of tuples
            List of (name, model) tuples for base models
        meta_model : estimator
            Model to use as meta-learner
        val_size : float
            Proportion of data to use for validation/blending
        random_state : int
            Random state for reproducibility
        use_features_in_meta : bool
            Whether to use original features along with base model predictions in meta-model
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.val_size = val_size
        self.random_state = random_state
        self.use_features_in_meta = use_features_in_meta
        self.base_models_fitted = []

    def fit(self, X, y):
        """
        Fit the blending ensemble.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Training features
        y : pandas.Series
            Target variable
        
        Returns:
        --------
        self
        """
        # split data into training and validation sets
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_size, random_state=self.random_state
        )

        self.base_models_fitted = []

        # fit base models on training data
        for name, model in self.base_models:
            logger.info(f'Fitting {name} on training data')
            fitted_model = model.fit(X_train, y_train)
            self.base_models_fitted.append((name, fitted_model))

        # generate predictions on validation data
        val_predictions = np.column_stack([
            model.predict(X_val) for _, model in self.base_models_fitted
        ])

        # prepare meta-features for meta-model
        if self.use_features_in_meta:
            # concatenate base model predictions with original features
            meta_features_df = pd.DataFrame(val_predictions, columns=[f'model_{i}' for i in range(len(self.base_models))])
            X_meta = pd.concat([meta_features_df, X_val.reset_index(drop=True)], axis=1)
        else:
            # use only base model predictions
            X_meta = pd.DataFrame(val_predictions, columns=[f'model_{i}' for i in range(len(self.base_models))])

        # fit meta-model on validation data
        logger.info('Fitting meta-model on validation data')
        self.meta_model.fit(X_meta, y_val)

        # refit base models on full dataset for future predictions
        self.base_models_fitted = []
        for name, model in self.base_models:
            logger.info(f'Refitting {name} on full dataset')
            fitted_model = model.fit(X, y)
            self.base_models_fitted.append((name, fitted_model))

        return self
    
    def predict(self, X):
        """
        Generate predictions for test data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Test features
        
        Returns:
        --------
        numpy.ndarray
            Predictions
        """
        # generate predictions from base models
        meta_features = np.column_stack([
            model.predict(X) for _, model in self.base_models_fitted
        ])

        # prepare meta-features for meta-model
        if self.use_features_in_meta:
            # concatenate base model predictions with original features
            meta_features_df = pd.DataFrame(meta_features, columns=[f'model_{i}' for i in range(len(self.base_models))])
            X_meta = pd.concat([meta_features_df, X.reset_index(drop=True)], axis=1)
        else:
            X_meta = pd.DataFrame(meta_features, columns=[f'model_{i}' for i in range(len(self.base_models))])

        # generate predictions from meta-model
        return self.meta_model.predict(X_meta)
    
def optimize_ensemble_weights(ensemble, X, y, n_trials=100, cv=5, random_state=42):
    """
    Optimize weights for a voting ensemble using randomized search.
    """
    logger.info("Optimizing ensemble weights")
    
    # Set up cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Random number generator
    rng = np.random.RandomState(random_state)
    
    best_rmse = float('inf')
    best_weights = None
    
    # Try different weight combinations
    for trial in range(n_trials):
        # Generate random weights
        weights = rng.rand(len(ensemble.models))
        # Normalize weights
        weights = weights / sum(weights)
        
        # Evaluate with cross-validation
        rmse_scores = []
        for train_idx, val_idx in kf.split(X):
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fit models
            models_fitted = []
            for name, model in ensemble.models:
                models_fitted.append((name, model.fit(X_train, y_train)))
            
            # Generate predictions
            predictions = np.column_stack([
                model.predict(X_val) for _, model in models_fitted
            ])
            
            # Weighted average - correctly apply weights across models
            # predictions shape: (n_samples, n_models)
            # weights shape: (n_models,)
            ensemble_pred = predictions @ weights  # Matrix multiplication
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
            rmse_scores.append(rmse)
        
        # Average RMSE across folds
        avg_rmse = np.mean(rmse_scores)
        
        # Check if this is the best so far
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_weights = weights
            logger.info(f"Trial {trial+1}/{n_trials}: Found better weights with RMSE {best_rmse:.2f}")
    
    logger.info(f"Best weights: {best_weights}, RMSE: {best_rmse:.2f}")
    return best_weights