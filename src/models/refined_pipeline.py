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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

class RefinedPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def run(self, data_path=os.path.join(project_root, 'data', 'processed')):
        print("Starting refined pipeline...")
        
        # 1. Load data
        train_data = pd.read_csv(os.path.join(data_path, 'train_processed.csv'))
        test_data = pd.read_csv(os.path.join(data_path, 'test_processed.csv'))
        
        # 2. Extract target and test IDs
        y_train = train_data['Annual Turnover'] 
        X_train = train_data.drop('Annual Turnover', axis=1)
        test_ids = test_data['Registration Number']
        X_test = test_data.drop('Registration Number', axis=1)
        
        # 3. Handle outliers
        X_train, y_train = self.handle_outliers(X_train, y_train)
        
        # 4. Try different target transformations
        y_train_transformed, transform_details = self.try_different_transformations(y_train)
        
        # 5. Select features using domain knowledge and importance
        selected_features = self.select_features_with_domain_knowledge(X_train, y_train_transformed)
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        # 6. Add carefully chosen interaction features
        X_train_enhanced, X_test_enhanced = self.add_targeted_interactions(
            X_train_selected, X_test_selected, selected_features
        )
        
        # 7. Standardize features
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
        optimized_models = self.optimize_hyperparameters(X_train_scaled, y_train_transformed)
        
        # 9. Generate enhanced stacked predictions
        final_preds_transformed = self.enhanced_stacking(
            X_train_scaled, y_train_transformed, X_test_scaled, optimized_models
        )
        
        # 10. Inverse transform predictions with safeguards
        if transform_details['name'] == 'log':
            final_preds = np.expm1(final_preds_transformed)
        elif transform_details['name'] == 'boxcox':
            final_preds = stats.inv_boxcox(final_preds_transformed, transform_details['lambda'])
        elif transform_details['name'] == 'sqrt':
            final_preds = np.square(final_preds_transformed)
        
        # 11. Apply range clipping based on training data
        final_preds = np.clip(
            final_preds,
            y_train.min() * 0.9,  # Allow 10% below minimum 
            y_train.max() * 1.1   # Allow 10% above maximum
        )
        
        # 13. Create submission
        submission = pd.DataFrame({
            'Registration Number': test_ids,
            'Annual Turnover': final_preds
        })
        
        # 14. Save submission with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        submission_path = os.path.join(project_root, 'data', 'submissions', f'refined_submission_{timestamp}.csv')
        submission.to_csv(submission_path, index=False)
        
        print(f"Submission saved to {submission_path}")
        print(f"Submission statistics - Min: {submission['Annual Turnover'].min():.2f}, "
              f"Max: {submission['Annual Turnover'].max():.2f}, "
              f"Mean: {submission['Annual Turnover'].mean():.2f}")
        
        return submission
    
    def extract_feature_importance(self, X_train, y_train_log):
        """Extract feature importance from multiple models"""
        
        # Train models for feature importance
        feature_models = {
            'xgb': xgb.XGBRegressor(n_estimators=100, random_state=self.random_state),
            'lgb': lgb.LGBMRegressor(n_estimators=100, random_state=self.random_state, verbose=-1)
        }
        
        importances = {}
        for name, model in feature_models.items():
            model.fit(X_train, y_train_log)
            if hasattr(model, 'feature_importances_'):
                imp = pd.Series(model.feature_importances_, index=X_train.columns)
                importances[name] = imp
        
        # Combine importance scores from different models
        combined_imp = pd.DataFrame(importances)
        mean_imp = combined_imp.mean(axis=1).sort_values(ascending=False)
        
        # Select top features (top 20)
        top_features = mean_imp.head(20).index.tolist()
        
        return top_features, mean_imp
    
    def add_targeted_interactions(self, X_train, X_test, top_features):
        """Add targeted interaction features based on domain knowledge"""
        
        X_train_enhanced = X_train.copy()
        X_test_enhanced = X_test.copy()
        
        # Define key pairs for interaction based on restaurant business logic
        interaction_pairs = [
            # Quality and popularity
            ('Restaurant Zomato Rating', 'Food Rating'),
            ('Hygiene Rating', 'Overall Restaurant Rating'),
            
            # Social media impact
            ('Facebook Popularity Quotient', 'Instagram Popularity Quotient'),
            
            # Restaurant age impact
            ('restaurant_age_years', 'Overall Restaurant Rating'),
            
            # Service quality
            ('Staff Responsivness', 'Service'),
            
            # Entertainment value
            ('offers_live_music', 'Lively')
        ]
        
        # Only create interactions if both features are available
        created_features = []
        for feat1, feat2 in interaction_pairs:
            if feat1 in X_train.columns and feat2 in X_train.columns:
                # Multiplication interaction
                feature_name = f'{feat1}_{feat2}_mult'
                X_train_enhanced[feature_name] = X_train[feat1] * X_train[feat2]
                X_test_enhanced[feature_name] = X_test[feat1] * X_test[feat2]
                created_features.append(feature_name)
        
        print(f"Created {len(created_features)} interaction features")
        return X_train_enhanced, X_test_enhanced
    
    def select_features_with_domain_knowledge(self, X_train, y_train_log, importance_threshold=0.01):
        """Select features based on both importance and domain knowledge"""
        
        # Calculate correlations with target
        correlations = X_train.apply(lambda x: x.corr(y_train_log) 
                                    if x.dtype.kind in 'ifc' else 0)
        
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
        feature_models = {
            'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lgb': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        }
        
        importances = {}
        for name, model in feature_models.items():
            model.fit(X_train, y_train_log)
            if hasattr(model, 'feature_importances_'):
                imp = pd.Series(model.feature_importances_, index=X_train.columns)
                importances[name] = imp
        
        # Combine importance scores
        combined_imp = pd.DataFrame(importances)
        mean_imp = combined_imp.mean(axis=1)
        
        # Select features above threshold
        important_features = mean_imp[mean_imp > importance_threshold].index.tolist()
        
        # Combine must-have and important features
        selected_features = list(set(must_have_features) | set(important_features))
        
        # Keep only features that exist in the dataset
        final_features = [f for f in selected_features if f in X_train.columns]
        
        print(f"Selected {len(final_features)} features using domain knowledge and importance")
        return final_features
    
    def optimize_hyperparameters(self, X_train, y_train_log):
        """Optimize hyperparameters for key models"""
        
        optimized_models = {}
        
        # Lasso
        lasso_params = {
            'alpha': [0.01, 0.05, 0.1, 0.5],
            'max_iter': [20000, 50000],
            'tol': [1e-4, 1e-5]
        }
        lasso_search = RandomizedSearchCV(
            Lasso(random_state=self.random_state), 
            lasso_params,
            n_iter=10,
            scoring='neg_root_mean_squared_error',
            cv=5,
            random_state=self.random_state
        )
        lasso_search.fit(X_train, y_train_log)
        optimized_models['lasso'] = lasso_search.best_estimator_
        print(f"Optimized Lasso - alpha: {lasso_search.best_estimator_.alpha:.6f}")
        
        # LightGBM
        lgb_params = {
            'learning_rate': [0.005, 0.01, 0.03, 0.05],
            'n_estimators': [200, 300, 500],
            'max_depth': [5, 6, 7, 8],
            'num_leaves': [15, 31, 63],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        lgb_search = RandomizedSearchCV(
            lgb.LGBMRegressor(random_state=self.random_state, verbose=-1), 
            lgb_params,
            n_iter=10,
            scoring='neg_root_mean_squared_error',
            cv=5,
            random_state=self.random_state
        )
        lgb_search.fit(X_train, y_train_log)
        optimized_models['lgb'] = lgb_search.best_estimator_
        print(f"Optimized LightGBM - params: {lgb_search.best_params_}")
        
        # XGBoost
        xgb_params = {
            'learning_rate': [0.005, 0.01, 0.02],
            'n_estimators': [400, 500, 600],
            'max_depth': [3, 4, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }
        xgb_search = RandomizedSearchCV(
            xgb.XGBRegressor(random_state=self.random_state), 
            xgb_params,
            n_iter=10,
            scoring='neg_root_mean_squared_error',
            cv=5,
            random_state=self.random_state
        )
        xgb_search.fit(X_train, y_train_log)
        optimized_models['xgb'] = xgb_search.best_estimator_
        print(f"Optimized XGBoost - params: {xgb_search.best_params_}")
        
        return optimized_models
    
    def stacked_predictions(self, X_train, y_train_log, X_test, models, random_state=42):
        """Generate stacked predictions using a meta-learner"""
        
        # Convert to numpy for faster operations
        X_train_values = X_train.values
        X_test_values = X_test.values
        
        # Cross-validation setup
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        meta_train = np.zeros((len(X_train), len(models)))
        meta_test = np.zeros((len(X_test), len(models)))
        
        # Generate out-of-fold predictions for meta-training
        for i, (name, model) in enumerate(models.items()):
            print(f"Generating meta-features for {name}...")
            test_preds = np.zeros(len(X_test))
            
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
            
            # Store test predictions
            meta_test[:, i] = test_preds
            
            # Also fit on full dataset
            model.fit(X_train_values, y_train_log)
        
        # Create meta-learner (simple Ridge regression)
        print("Training meta-learner...")
        meta_model = Ridge(alpha=0.5, random_state=random_state)
        
        # Train meta-learner
        meta_model.fit(meta_train, y_train_log)
        
        # Output meta-model coefficients (weights for each model)
        model_weights = {name: weight for name, weight in zip(models.keys(), meta_model.coef_)}
        print(f"Meta-model weights: {model_weights}")
        
        # Make final predictions
        final_preds = meta_model.predict(meta_test)
        
        return final_preds
    
    def enhanced_stacking(self, X_train, y_train_log, X_test, models, random_state=42):
        """Create enhanced stacked predictions with model selection"""
        
        # First evaluate each model individually with cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        model_rmse = {}
        
        for name, model in models.items():
            cv_scores = []
            for train_idx, val_idx in kf.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
                
                model_clone = clone(model)
                model_clone.fit(X_fold_train, y_fold_train)
                y_pred = model_clone.predict(X_fold_val)
                
                rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
                cv_scores.append(rmse)
            
            model_rmse[name] = np.mean(cv_scores)
            print(f"Model {name} - CV RMSE: {model_rmse[name]:.4f}")
        
        # Keep only the 3 best models for the final ensemble
        top_models = sorted(model_rmse.items(), key=lambda x: x[1])[:3]
        top_model_names = [name for name, _ in top_models]
        
        print(f"Using top {len(top_model_names)} models for ensemble: {top_model_names}")
        
        # Filter models
        selected_models = {name: models[name] for name in top_model_names}
        
        # Now create meta-features using only these models
        meta_train = np.zeros((len(X_train), len(selected_models)))
        meta_test = np.zeros((len(X_test), len(selected_models)))
        
        # Generate out-of-fold predictions
        for i, (name, model) in enumerate(selected_models.items()):
            print(f"Generating meta-features for {name}...")
            test_preds = np.zeros(len(X_test))
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train = y_train_log.iloc[train_idx]
                
                model_clone = clone(model)
                model_clone.fit(X_fold_train, y_fold_train)
                meta_train[val_idx, i] = model_clone.predict(X_fold_val)
                test_preds += model_clone.predict(X_test) / kf.n_splits
            
            meta_test[:, i] = test_preds
            model.fit(X_train, y_train_log)
        
        # Try both a simple average and a meta-model
        print("Training meta-model and computing simple average...")
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
        
        print(f"Meta-model CV RMSE: {meta_model_cv_score:.4f}")
        print(f"Simple average CV RMSE: {avg_model_cv_score:.4f}")
        
        # Return the better of the two approaches
        if meta_model_cv_score <= avg_model_cv_score:
            print("Using meta-model for final predictions")
            return meta_preds
        else:
            print("Using simple average for final predictions")
            return avg_preds
        
    def handle_outliers(self, X_train, y_train):
        """Identify and handle outliers in training data"""
        from scipy import stats
        
        # Calculate z-scores for the target
        z_scores = np.abs(stats.zscore(y_train))
        
        # Identify potential outliers (z-score > 3)
        outlier_indices = np.where(z_scores > 3)[0]
        
        if len(outlier_indices) > 0:
            print(f"Identified {len(outlier_indices)} potential outliers in target")
            
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
            
            # Cap outliers
            y_train_clean.loc[y_train < lower_bound] = lower_bound
            y_train_clean.loc[y_train > upper_bound] = upper_bound
            
            print(f"Applied capping to outliers using IQR method")
            return X_train, y_train_clean
    
        return X_train, y_train
    
    def try_different_transformations(self, y_train):
        """Try different transformations and select the best one"""
        from scipy import stats
        import numpy as np
        
        # Log transformation (current approach)
        y_log = np.log1p(y_train)
        
        # Box-Cox transformation (requires positive values)
        try:
            lambda_param, _ = stats.boxcox_normmax(y_train)
            y_boxcox = stats.boxcox(y_train, lambda_param)
            boxcox_details = {'transform': lambda x: stats.boxcox(x, lambda_param),
                            'inverse': lambda x: stats.inv_boxcox(x, lambda_param),
                            'lambda': lambda_param}
        except:
            y_boxcox = y_log
            boxcox_details = None
        
        # Square root transformation
        y_sqrt = np.sqrt(y_train)
        
        # Check normality of each transformation
        from scipy.stats import shapiro
        
        # Use a sample of 5000 points for large datasets
        if len(y_train) > 5000:
            idx = np.random.choice(len(y_train), 5000, replace=False)
            log_stat, log_p = shapiro(y_log.iloc[idx])
            boxcox_stat, boxcox_p = shapiro(y_boxcox[idx]) if boxcox_details else (0, 0)
            sqrt_stat, sqrt_p = shapiro(y_sqrt.iloc[idx])
        else:
            log_stat, log_p = shapiro(y_log)
            boxcox_stat, boxcox_p = shapiro(y_boxcox) if boxcox_details else (0, 0)
            sqrt_stat, sqrt_p = shapiro(y_sqrt)
        
        # Higher p-value indicates more normality
        p_values = {'log': log_p, 'boxcox': boxcox_p, 'sqrt': sqrt_p}
        best_transform = max(p_values, key=p_values.get)
        
        print(f"Transformation normality tests (p-values, higher is better):")
        print(f"Log: {log_p:.6f}")
        print(f"Box-Cox: {boxcox_p:.6f}")
        print(f"Square root: {sqrt_p:.6f}")
        print(f"Selected: {best_transform}")
        
        # Return the best transformation and details
        if best_transform == 'log':
            return y_log, {'transform': np.log1p, 'inverse': np.expm1, 'name': 'log'}
        elif best_transform == 'boxcox' and boxcox_details:
            return y_boxcox, boxcox_details
        else:
            return y_sqrt, {'transform': np.sqrt, 'inverse': np.square, 'name': 'sqrt'}

if __name__ == "__main__":
    pipeline = RefinedPipeline()
    submission = pipeline.run()