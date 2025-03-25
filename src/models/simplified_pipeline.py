import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

class SimplifiedPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def run(self, data_path=os.path.join(project_root, 'data', 'processed')):
        # Load data
        train_data = pd.read_csv(os.path.join(data_path, 'train_processed.csv'))
        test_data = pd.read_csv(os.path.join(data_path, 'test_processed.csv'))
        
        # Extract target and test IDs
        y_train = train_data['Annual Turnover']
        X_train = train_data.drop('Annual Turnover', axis=1)
        test_ids = test_data['Registration Number']
        X_test = test_data.drop('Registration Number', axis=1)
        
        # Ensure same columns
        common_cols = list(set(X_train.columns).intersection(X_test.columns))
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
        
        # Log-transform target (with safety)
        y_train_log = np.log1p(y_train)
        
        # Simple preprocessing - just standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to dataframe for feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Define simple ensemble of reliable models
        models = {
            'lasso': Lasso(alpha=0.001, max_iter=10000, random_state=self.random_state),
            'ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'elastic_net': ElasticNet(alpha=0.001, l1_ratio=0.7, max_iter=10000, random_state=self.random_state),
            'lgb': lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.01,
                max_depth=7,
                num_leaves=31,
                random_state=self.random_state,
                verbose=-1
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.01,
                max_depth=6,
                random_state=self.random_state
            )
        }
        
        # Cross-validation predictions for stacking
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        test_preds = np.zeros((len(X_test), len(models)))
        
        # Train each model with cross-validation
        for i, (name, model) in enumerate(models.items()):
            print(f"Training {name}...")
            # Full training for test predictions
            model.fit(X_train_scaled, y_train_log)
            
            # Make test predictions
            test_preds[:, i] = model.predict(X_test_scaled)
        
        # Simple averaging of predictions
        final_preds_log = np.mean(test_preds, axis=1)
        
        # Inverse transform with safeguards
        min_val = np.log1p(y_train.min())
        max_val = np.log1p(y_train.max())
        
        # Clip predictions to the range seen in training data
        final_preds_log = np.clip(final_preds_log, min_val, max_val)
        
        # Inverse transform
        final_preds = np.expm1(final_preds_log)
        
        # Create submission
        submission = pd.DataFrame({
            'Registration Number': test_ids,
            'Annual Turnover': final_preds
        })
        
        return submission

if __name__ == "__main__":
    pipeline = SimplifiedPipeline()
    submission = pipeline.run()
    submission.to_csv(os.path.join(project_root, 'data', 'submissions', 'simple_ensemble_submission.csv'), index=False)
    print(f"Submission statistics - Min: {submission['Annual Turnover'].min():.2f}, "
          f"Max: {submission['Annual Turnover'].max():.2f}, "
          f"Mean: {submission['Annual Turnover'].mean():.2f}")