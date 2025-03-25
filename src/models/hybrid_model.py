# src/models/hybrid_model.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger('hybrid_model')

class HybridModel(BaseEstimator, RegressorMixin):
    """
    A hybrid model that combines linear and non-linear components
    with specialized error correction.
    
    The model works in three stages:
    1. A linear model captures basic relationships
    2. A non-linear model captures complex patterns in the residuals
    3. An error correction model addresses systematic prediction errors
    """
    
    def __init__(self, 
                 linear_model=None,
                 nonlinear_model=None,
                 error_model=None,
                 alpha=0.5):
        """
        Initialize the hybrid model.
        
        Parameters:
        -----------
        linear_model : estimator, optional
            Linear model component (default: ElasticNet)
        nonlinear_model : estimator, optional
            Non-linear model component (default: LightGBM)
        error_model : estimator, optional
            Error correction model (default: XGBoost)
        alpha : float
            Weight for combining linear and non-linear predictions (default: 0.5)
        """
        self.linear_model = linear_model or ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=2000)
        self.nonlinear_model = nonlinear_model or lgb.LGBMRegressor(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8
        )
        self.error_model = error_model or xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8
        )
        self.alpha = alpha
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the hybrid model.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Training features
        y : pandas.Series or numpy.ndarray
            Target variable
        sample_weight : numpy.ndarray, optional
            Sample weights for training
            
        Returns:
        --------
        self
        """
        logger.info("Fitting hybrid model")
        
        # Convert to pandas DataFrame/Series if numpy arrays
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
            
        # Reset indices to ensure alignment
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Store original feature names
        self.feature_names_ = X.columns.tolist()
        
        # 1. Train linear model
        logger.info("Training linear component")
        if sample_weight is not None:
            self.linear_model.fit(X, y, sample_weight=sample_weight)
        else:
            self.linear_model.fit(X, y)
            
        # Get linear predictions
        linear_preds = self.linear_model.predict(X)
        
        # 2. Train non-linear model on residuals
        logger.info("Training non-linear component on residuals")
        residuals = y - linear_preds
        
        if sample_weight is not None:
            self.nonlinear_model.fit(X, residuals, sample_weight=sample_weight)
        else:
            self.nonlinear_model.fit(X, residuals)
            
        # Get non-linear predictions (of residuals)
        nonlinear_preds = self.nonlinear_model.predict(X)
        
        # 3. Create combined predictions
        combined_preds = (self.alpha * linear_preds) + ((1-self.alpha) * nonlinear_preds)
        
        # 4. Create error features for correction model
        logger.info("Training error correction component")
        abs_errors = np.abs(y - combined_preds)
        squared_errors = (y - combined_preds) ** 2
        
        # Create error features DataFrame with the same index as X
        error_features = pd.DataFrame({
            'predicted_value': combined_preds,
            'predicted_abs_error': abs_errors,
            'predicted_squared_error': squared_errors
        }, index=X.index)
        
        # Add top feature interactions (only if the feature exists in X)
        if hasattr(self.nonlinear_model, 'feature_importances_'):
            feature_imp = pd.Series(
                self.nonlinear_model.feature_importances_,
                index=self.feature_names_
            ).sort_values(ascending=False)
            
            top_features = feature_imp.index[:min(5, len(feature_imp))].tolist()
            
            for feat in top_features:
                if feat in X.columns:
                    error_features[f'pred_x_{feat}'] = combined_preds * X[feat]
        
        # Combine with original features - make a copy to avoid modifying original
        X_error = pd.concat([X, error_features], axis=1)
        
        # Make sure X_error and final_residuals have the same shape
        final_residuals = y - combined_preds
        
        # Verify shapes match
        if len(X_error) != len(final_residuals):
            logger.warning(f"Shape mismatch: X_error={X_error.shape}, final_residuals={final_residuals.shape}")
            # Fix the issue by aligning indices
            common_idx = X_error.index.intersection(final_residuals.index)
            X_error = X_error.loc[common_idx]
            final_residuals = final_residuals.loc[common_idx]
            
        # Check for NaN values
        if X_error.isna().any().any():
            logger.warning(f"Found {X_error.isna().sum().sum()} NaN values in X_error. Filling with 0.")
            X_error = X_error.fillna(0)
            
        if final_residuals.isna().any():
            logger.warning(f"Found {final_residuals.isna().sum()} NaN values in final_residuals. Filling with 0.")
            final_residuals = final_residuals.fillna(0)
            
        # Train error correction model (without sample weights for simplicity)
        self.error_model.fit(X_error, final_residuals)
        
        # Save shapes for debugging
        self.X_error_shape_ = X_error.shape
        self.final_residuals_shape_ = (len(final_residuals),)
        
        # Calculate and log training metrics
        final_predictions = self._get_final_predictions(X)
        mse = mean_squared_error(y, final_predictions)
        rmse = np.sqrt(mse)
        logger.info(f"Hybrid model training complete - Training RMSE: {rmse:.4f}")
        
        return self
    
    def _get_final_predictions(self, X):
        """
        Generate final predictions by combining all components.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature data
            
        Returns:
        --------
        numpy.ndarray
            Final predictions
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_)
            
        # Reset index for consistency
        X = X.reset_index(drop=True)
        
        # Get predictions from each component
        linear_preds = self.linear_model.predict(X)
        nonlinear_preds = self.nonlinear_model.predict(X)
        
        # Combined predictions (weighted)
        combined_preds = (self.alpha * linear_preds) + ((1-self.alpha) * nonlinear_preds)
        
        # Create error features with same index as X
        error_features = pd.DataFrame({
            'predicted_value': combined_preds,
            'predicted_abs_error': np.zeros_like(combined_preds),  # Placeholder
            'predicted_squared_error': np.zeros_like(combined_preds)  # Placeholder
        }, index=X.index)
        
        # Add interaction features if we have feature importances
        if hasattr(self.nonlinear_model, 'feature_importances_'):
            feature_imp = pd.Series(
                self.nonlinear_model.feature_importances_,
                index=self.feature_names_
            ).sort_values(ascending=False)
            
            top_features = feature_imp.index[:min(5, len(feature_imp))].tolist()
            
            for feat in top_features:
                if feat in X.columns:
                    error_features[f'pred_x_{feat}'] = combined_preds * X[feat]
        
        # Combine with original features
        X_error = pd.concat([X, error_features], axis=1)
        
        # Fill any NaN values
        X_error = X_error.fillna(0)
        
        # Get error correction
        corrections = self.error_model.predict(X_error)
        
        # Final predictions
        final_predictions = combined_preds + corrections
        
        return final_predictions
    
    def predict(self, X):
        """
        Generate predictions for new data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Feature data
            
        Returns:
        --------
        numpy.ndarray
            Predictions
        """
        return self._get_final_predictions(X)
    
    def get_feature_importances(self):
        """
        Get combined feature importances from all components.
        
        Returns:
        --------
        pandas.DataFrame
            Feature importances from each component
        """
        importances = pd.DataFrame(index=self.feature_names_)
        
        # Extract linear coefficients
        if hasattr(self.linear_model, 'coef_'):
            linear_coef = np.abs(self.linear_model.coef_)
            linear_imp = linear_coef / np.sum(linear_coef) if np.sum(linear_coef) > 0 else linear_coef
            importances['linear'] = linear_imp
            
        # Extract non-linear feature importances
        if hasattr(self.nonlinear_model, 'feature_importances_'):
            importances['nonlinear'] = self.nonlinear_model.feature_importances_
            
        # Calculate combined importance
        if 'linear' in importances.columns and 'nonlinear' in importances.columns:
            importances['combined'] = self.alpha * importances['linear'] + (1 - self.alpha) * importances['nonlinear']
            importances = importances.sort_values('combined', ascending=False)
            
        return importances