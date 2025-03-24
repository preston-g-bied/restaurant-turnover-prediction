# src/features/feature_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
import category_encoders as ce
from datetime import datetime
import logging

class FeatureProcessor:
    """
    A class to handle all feature engineering steps with proper fit/transform separation.
    This ensures consistency between training and test data processing.
    """
    
    def __init__(self, target_column='Annual Turnover', log_level=logging.INFO):
        """
        Initialize the feature processor with empty transformers.
        
        Parameters:
        -----------
        target_column : str
            Name of the target column to exclude from transformations
        log_level : int
            Logging level (e.g., logging.INFO, logging.DEBUG)
        """
        # Set up logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('FeatureProcessor')
        
        # Store target column name
        self.target_column = target_column
        
        # Imputers
        self.numeric_imputer = None
        
        # Encoders
        self.target_encoders = {}
        self.label_encoders = {}
        self.onehot_encoders = {}
        
        # Feature selectors
        self.feature_selector = None
        self.selected_features = None
        
        # Feature lists
        self.high_cardinality_cols = []
        self.low_cardinality_cols = []
        self.binary_cols = []
        
        # State tracking
        self.is_fitted = False
        
    def fit_transform(self, X, y=None):
        """
        Fit all transformers and transform the input data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with features and possibly target
        y : pandas.Series, optional
            Target variable if not included in X
        
        Returns:
        --------
        pandas.DataFrame
            Transformed features
        """
        self.logger.info("Starting fit_transform process")
        
        # Extract target if it's in X and not provided separately
        if y is None and self.target_column in X.columns:
            self.logger.info(f"Extracting target column '{self.target_column}' from input data")
            y = X[self.target_column].copy()
            X = X.drop(columns=[self.target_column])
            
        # Store original columns for verification
        self.original_columns = X.columns.tolist()
        self.logger.info(f"Original feature count: {len(self.original_columns)}")
        
        # Handle missing values
        X_processed = self._fit_handle_missing_values(X)
        
        # Extract temporal features
        X_processed = self._extract_temporal_features(X_processed)
        
        # Transform numeric features
        X_processed = self._fit_transform_numeric_features(X_processed)
        
        # Identify categorical columns by type
        self._identify_categorical_columns(X_processed)
        
        # Encode categorical features
        X_processed = self._fit_encode_categorical_features(X_processed, y)
        
        # Create interaction features
        X_processed = self._create_interaction_features(X_processed)
        
        # Feature selection (if y is provided)
        if y is not None:
            X_processed, self.selected_features = self._fit_select_features(X_processed, y)
        
        self.is_fitted = True
        self.logger.info(f"Fit_transform complete. Final feature count: {X_processed.shape[1]}")
        
        return X_processed
    
    def transform(self, X, apply_selection=True):
        """
        Transform new data using already fitted transformers.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data to transform
        apply_selection : bool
            Whether to apply feature selection
        
        Returns:
        --------
        pandas.DataFrame
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("FeatureProcessor has not been fitted yet. Call fit_transform first.")
        
        self.logger.info("Starting transform process")
        
        # Make sure target column is not in the test data
        if self.target_column in X.columns:
            self.logger.warning(f"Removing target column '{self.target_column}' from test data")
            X = X.drop(columns=[self.target_column])
        
        # Handle missing values
        X_processed = self._transform_handle_missing_values(X)
        
        # Extract temporal features
        X_processed = self._extract_temporal_features(X_processed)
        
        # Transform numeric features
        X_processed = self._transform_numeric_features(X_processed)
        
        # Encode categorical features
        X_processed = self._transform_encode_categorical_features(X_processed)
        
        # Create interaction features
        X_processed = self._create_interaction_features(X_processed)
        
        # Apply feature selection
        if apply_selection and self.selected_features is not None:
            # Get only features that exist in the processed data
            available_features = [f for f in self.selected_features if f in X_processed.columns]
            if len(available_features) < len(self.selected_features):
                missing_features = set(self.selected_features) - set(available_features)
                self.logger.warning(
                    f"Missing {len(missing_features)} selected features in transform data: {missing_features}"
                )
            X_processed = X_processed[available_features]
        
        self.logger.info(f"Transform complete. Final feature count: {X_processed.shape[1]}")
        return X_processed
    
    def _identify_categorical_columns(self, X):
        """Identify and categorize categorical columns based on cardinality."""
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Reset lists to prevent duplicate entries if called multiple times
        self.binary_cols = []
        self.low_cardinality_cols = []
        self.high_cardinality_cols = []
        
        for col in categorical_cols:
            # Skip target column if present
            if col == self.target_column:
                continue
                
            n_unique = X[col].nunique()
            if n_unique <= 2:
                self.binary_cols.append(col)
            elif n_unique <= 15:
                self.low_cardinality_cols.append(col)
            else:
                self.high_cardinality_cols.append(col)
        
        self.logger.info(f"Identified {len(self.binary_cols)} binary columns")
        self.logger.info(f"Identified {len(self.low_cardinality_cols)} low cardinality columns")
        self.logger.info(f"Identified {len(self.high_cardinality_cols)} high cardinality columns")
    
    def _fit_handle_missing_values(self, X):
        """
        Fit imputers and handle missing values.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data to fit imputers on and transform
            
        Returns:
        --------
        pandas.DataFrame
            Data with missing values handled
        """
        self.logger.info("Fitting missing value handlers")
        X_imputed = X.copy()
        
        # Create missing indicators for features with high missingness (>20%)
        high_missingness = ['Live Sports Rating', 'Value Deals Rating', 
                           'Comedy Gigs Rating', 'Live Music Rating']
        
        for col in high_missingness:
            if col in X.columns:
                X_imputed[f'has_{col.lower().replace(" ", "_")}'] = X[col].notna().astype(int)
        
        # Fit imputer for numeric features - EXCLUDE target column
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        # Make sure the target is not in numeric features
        if self.target_column in numeric_features:
            numeric_features = numeric_features.drop(self.target_column)
            
        self.numeric_imputer = SimpleImputer(strategy='median')
        X_imputed[numeric_features] = self.numeric_imputer.fit_transform(X[numeric_features])
        
        # Store the features the imputer was trained on
        self.imputer_features = numeric_features.tolist()
        
        return X_imputed
    
    def _transform_handle_missing_values(self, X):
        """
        Transform new data using fitted imputers.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data to transform
            
        Returns:
        --------
        pandas.DataFrame
            Data with missing values handled
        """
        self.logger.info("Applying missing value handlers")
        X_imputed = X.copy()
        
        # Create missing indicators
        high_missingness = ['Live Sports Rating', 'Value Deals Rating', 
                           'Comedy Gigs Rating', 'Live Music Rating']
        
        for col in high_missingness:
            if col in X.columns:
                X_imputed[f'has_{col.lower().replace(" ", "_")}'] = X[col].notna().astype(int)
        
        # Apply imputer to numeric features that exist in this dataset
        if self.numeric_imputer is not None:
            # Find common features between what the imputer was fitted on and current data
            available_features = [f for f in self.imputer_features if f in X.columns]
            
            if len(available_features) > 0:
                self.logger.info(f"Imputing {len(available_features)} numeric features")
                X_imputed[available_features] = self.numeric_imputer.transform(X[available_features])
            else:
                self.logger.warning("No numeric features to impute")
        
        return X_imputed
    
    def _extract_temporal_features(self, X):
        """
        Extract temporal features from date columns.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with date column
            
        Returns:
        --------
        pandas.DataFrame
            Data with additional temporal features
        """
        self.logger.info("Extracting temporal features")
        X_temporal = X.copy()
        date_col = 'Opening Day of Restaurant'
        
        if date_col in X.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(X[date_col]):
                X_temporal[date_col] = pd.to_datetime(
                    X_temporal[date_col], format='%d-%m-%Y', errors='coerce'
                )
            
            # Current date for age calculation
            current_date = datetime.now()
            
            # Create temporal features
            X_temporal['restaurant_age_days'] = (current_date - X_temporal[date_col]).dt.days
            X_temporal['restaurant_age_years'] = X_temporal['restaurant_age_days'] / 365.25
            X_temporal['opening_year'] = X_temporal[date_col].dt.year
            X_temporal['opening_month'] = X_temporal[date_col].dt.month
            X_temporal['opening_day'] = X_temporal[date_col].dt.day
            X_temporal['opening_day_of_week'] = X_temporal[date_col].dt.dayofweek
            X_temporal['opening_quarter'] = X_temporal[date_col].dt.quarter
            
            # Create seasonal indicators
            season_map = {
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            }
            X_temporal['opening_season'] = X_temporal['opening_month'].map(season_map)
            
            # Drop original date column
            X_temporal = X_temporal.drop(columns=[date_col])
        
        return X_temporal
    
    def _fit_transform_numeric_features(self, X):
        """
        Fit and transform numeric features (log transform).
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with numeric features
            
        Returns:
        --------
        pandas.DataFrame
            Data with transformed numeric features
        """
        self.logger.info("Fitting and transforming numeric features")
        X_transformed = X.copy()
        
        # Log transform applicable columns
        log_transform_cols = [
            'Facebook Popularity Quotient',
            'Instagram Popularity Quotient',
            'restaurant_age_days',
            'Live Music Rating',
            'Comedy Gigs Rating',
            'Value Deals Rating',
            'Live Sports Rating'
        ]
        
        # Only add target to log transform if we're actually processing it
        if self.target_column in X.columns:
            log_transform_cols.append(self.target_column)
        
        for col in log_transform_cols:
            if col in X.columns:
                # Add a small constant to avoid log(0)
                X_transformed[f'{col}_log'] = np.log1p(X_transformed[col])
        
        return X_transformed
    
    def _transform_numeric_features(self, X):
        """
        Transform numeric features on new data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with numeric features
            
        Returns:
        --------
        pandas.DataFrame
            Data with transformed numeric features
        """
        self.logger.info("Transforming numeric features")
        X_transformed = X.copy()
        
        # Log transform applicable columns - target should be EXCLUDED here
        log_transform_cols = [
            'Facebook Popularity Quotient',
            'Instagram Popularity Quotient',
            'restaurant_age_days',
            'Live Music Rating',
            'Comedy Gigs Rating',
            'Value Deals Rating',
            'Live Sports Rating'
        ]
        
        for col in log_transform_cols:
            if col in X.columns:
                # Add a small constant to avoid log(0)
                X_transformed[f'{col}_log'] = np.log1p(X_transformed[col])
        
        return X_transformed
    
    def _fit_encode_categorical_features(self, X, y=None):
        """
        Fit encoders and transform categorical features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with categorical features
        y : pandas.Series, optional
            Target variable for target encoding
            
        Returns:
        --------
        pandas.DataFrame
            Data with encoded categorical features
        """
        self.logger.info("Fitting and transforming categorical features")
        X_encoded = X.copy()
        
        # Standardize city names if City column exists
        if 'City' in X_encoded.columns:
            from src.features.build_features import city_mapping
            X_encoded['City'] = X_encoded['City'].str.lower().map(
                lambda x: city_mapping.get(x, x)
            )
        
        # 1. Encode binary features
        for col in self.binary_cols:
            if col in X_encoded.columns:
                self.label_encoders[col] = LabelEncoder()
                X_encoded[col] = self.label_encoders[col].fit_transform(X_encoded[col])
        
        # 2. Encode low cardinality features with one-hot encoding
        for col in self.low_cardinality_cols:
            if col in X_encoded.columns:
                self.onehot_encoders[col] = OneHotEncoder(
                    sparse_output=False, drop='first', handle_unknown='ignore'
                )
                encoded = self.onehot_encoders[col].fit_transform(X_encoded[[col]])
                
                # Create column names
                encoded_cols = [f'{col}_{cat}' for cat in 
                               self.onehot_encoders[col].categories_[0][1:]]
                
                # Add encoded columns to dataframe
                for i, encoded_col in enumerate(encoded_cols):
                    X_encoded[encoded_col] = encoded[:, i]
                
                # Drop original column
                X_encoded = X_encoded.drop(col, axis=1)
        
        # 3. Target encode high cardinality features if y is provided
        if y is not None:
            for col in self.high_cardinality_cols:
                if col in X_encoded.columns:
                    self.target_encoders[col] = ce.TargetEncoder()
                    X_encoded[f'{col}_encoded'] = self.target_encoders[col].fit_transform(
                        X_encoded[col], y
                    )
                    # Drop original column
                    X_encoded = X_encoded.drop(col, axis=1)
        else:
            # If no target is provided, use label encoding as fallback
            for col in self.high_cardinality_cols:
                if col in X_encoded.columns:
                    self.label_encoders[col] = LabelEncoder()
                    X_encoded[col] = self.label_encoders[col].fit_transform(X_encoded[col])
        
        return X_encoded
    
    def _transform_encode_categorical_features(self, X):
        """
        Transform categorical features on new data using fitted encoders.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with categorical features
            
        Returns:
        --------
        pandas.DataFrame
            Data with encoded categorical features
        """
        self.logger.info("Transforming categorical features")
        X_encoded = X.copy()
        
        # Standardize city names if City column exists
        if 'City' in X_encoded.columns:
            from src.features.build_features import city_mapping
            X_encoded['City'] = X_encoded['City'].str.lower().map(
                lambda x: city_mapping.get(x, x)
            )
        
        # 1. Transform binary features
        for col, encoder in self.label_encoders.items():
            if col in X_encoded.columns:
                try:
                    # Handle potential new categories
                    X_encoded[col] = X_encoded[col].map(
                        lambda x: x if x in encoder.classes_ else encoder.classes_[0]
                    )
                    X_encoded[col] = encoder.transform(X_encoded[col])
                except Exception as e:
                    self.logger.error(f"Error transforming column {col}: {str(e)}")
        
        # 2. Transform low cardinality features with one-hot encoding
        for col, encoder in self.onehot_encoders.items():
            if col in X_encoded.columns:
                try:
                    encoded = encoder.transform(X_encoded[[col]])
                    
                    # Create column names
                    encoded_cols = [f'{col}_{cat}' for cat in encoder.categories_[0][1:]]
                    
                    # Add encoded columns to dataframe
                    for i, encoded_col in enumerate(encoded_cols):
                        X_encoded[encoded_col] = encoded[:, i]
                    
                    # Drop original column
                    X_encoded = X_encoded.drop(col, axis=1)
                except Exception as e:
                    self.logger.error(f"Error one-hot encoding column {col}: {str(e)}")
        
        # 3. Transform high cardinality features with target encoding
        for col, encoder in self.target_encoders.items():
            if col in X_encoded.columns:
                try:
                    X_encoded[f'{col}_encoded'] = encoder.transform(X_encoded[col])
                    # Drop original column
                    X_encoded = X_encoded.drop(col, axis=1)
                except Exception as e:
                    self.logger.error(f"Error target encoding column {col}: {str(e)}")
        
        return X_encoded
    
    def _create_interaction_features(self, X):
        """
        Create interaction features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame
            Data with additional interaction features
        """
        self.logger.info("Creating interaction features")
        X_with_interactions = X.copy()
        
        # Define interactions based on domain knowledge
        interactions = [
            # Quality and popularity interactions
            ('Restaurant Zomato Rating', 'Food Rating'),
            ('Overall Restaurant Rating', 'Facebook Popularity Quotient'),
            ('Overall Restaurant Rating', 'Instagram Popularity Quotient'),
            ('Hygiene Rating', 'Food Rating'),
            
            # Experience-based interactions
            ('Ambience', 'Service'),
            ('Lively', 'Privacy'),
            ('Order Wait Time', 'Food Rating'),
        ]
        
        # Create multiplicative interactions for features that exist
        for col1, col2 in interactions:
            if col1 in X.columns and col2 in X.columns:
                X_with_interactions[f'{col1}_{col2}_interaction'] = X[col1] * X[col2]
        
        # Create ratio features
        ratio_features = [
            ('Food Rating', 'Value for Money', 'premium_factor'),
            ('Order Wait Time', 'Staff Responsivness', 'service_efficiency'),
            ('Instagram Popularity Quotient', 'Facebook Popularity Quotient', 'social_media_balance'),
            ('Restaurant Zomato Rating', 'Overall Restaurant Rating', 'rating_discrepancy'),
            ('Food Rating', 'Hygiene Rating', 'quality_cleanliness_balance'),
            ('Lively', 'Comfortablility', 'energy_comfort_balance')
        ]
        
        for col1, col2, name in ratio_features:
            if col1 in X.columns and col2 in X.columns:
                # Avoid division by 0
                X_with_interactions[name] = X[col1] / X[col2].replace(0, 0.001)
        
        return X_with_interactions
    
    def _fit_select_features(self, X, y, k=20):
        """
        Fit feature selector and select top k features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series
            Target variable
        k : int
            Number of top features to select
            
        Returns:
        --------
        pandas.DataFrame, list
            Selected features dataframe and list of feature names
        """
        self.logger.info(f"Selecting top {k} features")
        
        # Create selector
        self.feature_selector = SelectKBest(f_regression, k=k)
        
        # Fit and transform
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        self.logger.info(f"Selected features: {selected_features}")
        
        # Return selected features dataframe and feature names
        return pd.DataFrame(X_selected, columns=selected_features), selected_features