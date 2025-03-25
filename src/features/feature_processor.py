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
        
        # Handle missing values with enhanced method
        X_processed = self._handle_missing_values_enhanced(X)
        X_processed = self._fit_handle_missing_values(X_processed)

        # process entertainment features
        X_processed = self._process_entertainment_features(X_processed)
        
        # Extract temporal features (basic and advanced)
        X_processed = self._extract_temporal_features(X_processed)
        X_processed = self._extract_advanced_temporal_features(X_processed)
        
        # Transform numeric features
        X_processed = self._fit_transform_numeric_features(X_processed)
        
        # Identify categorical columns by type
        self._identify_categorical_columns(X_processed)
        
        # Encode categorical features
        X_processed = self._fit_encode_categorical_features(X_processed, y)
        
        # Apply new feature engineering steps
        X_processed = self._create_enhanced_rating_features(X_processed)
        X_processed = self._enhance_location_features(X_processed)
        
        # Create interaction features (original method)
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
        """
        if not self.is_fitted:
            raise ValueError("FeatureProcessor has not been fitted yet. Call fit_transform first.")
        
        self.logger.info("Starting transform process")
        
        # Make sure target column is not in the test data
        if self.target_column in X.columns:
            self.logger.warning(f"Removing target column '{self.target_column}' from test data")
            X = X.drop(columns=[self.target_column])
        
        # Handle missing values with enhanced method
        X_processed = self._handle_missing_values_enhanced(X)
        X_processed = self._transform_handle_missing_values(X_processed)

        # process entertainment features
        X_processed = self._process_entertainment_features(X_processed)
        
        # Extract temporal features (basic and advanced)
        X_processed = self._extract_temporal_features(X_processed)
        X_processed = self._extract_advanced_temporal_features(X_processed)
        
        # Transform numeric features
        X_processed = self._transform_numeric_features(X_processed)
        
        # Encode categorical features
        X_processed = self._transform_encode_categorical_features(X_processed)
        
        # Apply new feature engineering steps
        X_processed = self._create_enhanced_rating_features(X_processed)
        X_processed = self._enhance_location_features(X_processed)
        
        # Create interaction features (original method)
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
    
    def _handle_missing_values_enhanced(self, X):
        """
        Enhanced missing value handling that adds indicator features but doesn't modify imputation.
        """
        self.logger.info("Creating enhanced missing value indicators")
        X_enhanced = X.copy()
        
        # Define entertainment columns
        entertainment_cols = ['Live Music Rating', 'Comedy Gigs Rating', 'Value Deals Rating', 'Live Sports Rating']
        present_ent_cols = [col for col in entertainment_cols if col in X.columns]
        
        # Create binary indicators for feature presence with more meaningful names
        for col in present_ent_cols:
            feat_name = col.replace('Rating', '').strip().lower().replace(' ', '_')
            X_enhanced[f'offers_{feat_name}'] = X[col].notna().astype(int)
        
        # Count how many entertainment options the restaurant offers
        if present_ent_cols:
            X_enhanced['entertainment_options_count'] = sum(X[col].notna() for col in present_ent_cols)
        
        return X_enhanced
    
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
    
    def _extract_advanced_temporal_features(self, X):
        """
        Extract advanced temporal features including age buckets and seasonality.
    
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with basic temporal features
            
        Returns:
        --------
        pandas.DataFrame
            Data with enhanced temporal features
        """
        self.logger.info("Extracting advanced temporal features")
        X_temporal = X.copy()

        # Skip if we don't have the restaurant age
        if 'restaurant_age_years' not in X_temporal.columns:
            self.logger.warning("restaurant_age_years not found, skipping advanced temporal features")
            return X_temporal
        
        # 1. Create age buckets (non-linear relationship with turnover)
        age_bins = [0, 1, 3, 5, 10, 20, 100]
        age_labels = ['0-1 years', '1-3 years', '3-5 years', '5-10 years', '10-20 years', '20+ years']
        
        X_temporal['age_bucket'] = pd.cut(
            X_temporal['restaurant_age_years'], 
            bins=age_bins, 
            labels=age_labels,
            include_lowest=True
        )

        # Convert to one-hot encoding
        age_dummies = pd.get_dummies(X_temporal['age_bucket'], prefix='age')
        X_temporal = pd.concat([X_temporal, age_dummies], axis=1)
        X_temporal.drop('age_bucket', axis=1, inplace=True)
        
        # 2. Restaurant maturity effect (diminishing returns)
        X_temporal['maturity_factor'] = 1 - np.exp(-X_temporal['restaurant_age_years'] / 5)
        
        # 3. Opening month seasonality
        if 'opening_month' in X_temporal.columns:
            # High season (typically November-January, and summer months)
            X_temporal['opened_high_season'] = X_temporal['opening_month'].isin([11, 12, 1, 6, 7, 8]).astype(int)
            
            # Festival/holiday season opening (typically end of year)
            X_temporal['opened_festival_season'] = X_temporal['opening_month'].isin([11, 12]).astype(int)
            
            # Create quarter features (already done in your code via opening_quarter)
            
        # 4. Day of week effect
        if 'opening_day_of_week' in X_temporal.columns:
            # Weekend vs weekday opening
            X_temporal['opened_weekend'] = X_temporal['opening_day_of_week'].isin([5, 6]).astype(int)
        
        # 5. Key interactions between age and other features
        if 'maturity_factor' in X_temporal.columns:
            for col in ['Hygiene Rating', 'Food Rating', 'Restaurant Zomato Rating']:
                if col in X_temporal.columns:
                    # Mature restaurants might have more reliable ratings
                    X_temporal[f'mature_{col.lower().replace(" ", "_")}'] = X_temporal['maturity_factor'] * X_temporal[col]
        
        return X_temporal
    
    def _enhance_location_features(self, X):
        """
        Create advanced location-based features including location type interactions.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
                
        Returns:
        --------
        pandas.DataFrame
            Data with enhanced location features
        """
        self.logger.info("Enhancing location features")
        X_location = X.copy()
        
        # 1. Create city tier groupings (if City_encoded exists)
        if 'City_encoded' in X_location.columns:
            # Our City has already been encoded to a numerical value
            # We can use the encoded value to create tier features
            X_location['major_city_indicator'] = (X_location['City_encoded'] > X_location['City_encoded'].median()).astype(int)
        
        # 2. Create interactions between location type and other features
        location_cols = [col for col in X_location.columns if 'Restaurant Location' in col]
        if location_cols:
            # For party hubs - entertainment and liveliness likely more important
            if 'Restaurant Location_Near Party Hub' in X_location.columns:
                party_hub = X_location['Restaurant Location_Near Party Hub']
                
                # Social media might have different impact at party hubs
                if 'Facebook Popularity Quotient' in X_location.columns:
                    X_location['party_hub_social_impact'] = party_hub * X_location['Facebook Popularity Quotient']
                
                # Lively rating likely more important at party hubs
                if 'Lively' in X_location.columns:
                    X_location['party_hub_lively_effect'] = party_hub * X_location['Lively']
                    
                # Entertainment options might be more valuable at party hubs
                if 'entertainment_options_count' in X_location.columns:
                    X_location['party_hub_entertainment_effect'] = party_hub * X_location['entertainment_options_count']
            
            # For business hubs - food quality and service likely more important
            if 'Restaurant Location_Near Business Hub' in X_location.columns or 'Restaurant Location_Near Party Hub' in X_location.columns:
                # Infer business hub as opposite of party hub if needed
                if 'Restaurant Location_Near Business Hub' in X_location.columns:
                    business_hub = X_location['Restaurant Location_Near Business Hub']
                else:
                    business_hub = 1 - X_location['Restaurant Location_Near Party Hub']
                
                # Food rating likely more important at business hubs
                if 'Food Rating' in X_location.columns:
                    X_location['business_hub_food_effect'] = business_hub * X_location['Food Rating']
                
                # Service likely more important at business hubs
                if 'Service' in X_location.columns:
                    X_location['business_hub_service_effect'] = business_hub * X_location['Service']
        
        # 3. Combine Restaurant City Tier with location type
        if 'Restaurant City Tier' in X_location.columns:
            for loc_col in location_cols:
                X_location[f'{loc_col}_city_tier_effect'] = X_location[loc_col] * X_location['Restaurant City Tier']
        
        return X_location
    
    def _process_entertainment_features(self, X):
        """
        Process entertainment features after imputation.
        """
        self.logger.info("Processing entertainment features")
        X_processed = X.copy()
        
        # Define entertainment columns
        entertainment_cols = ['Live Music Rating', 'Comedy Gigs Rating', 'Value Deals Rating', 'Live Sports Rating']
        present_ent_cols = [col for col in entertainment_cols if col in X.columns]
        
        # Set imputed values to 0 for restaurants that don't offer the feature
        for col in present_ent_cols:
            if col in X.columns:
                indicator = f'offers_{col.replace("Rating", "").strip().lower().replace(" ", "_")}'
                if indicator in X.columns:
                    # Keep original values where the feature exists
                    # Set to 0 where the restaurant doesn't offer the feature
                    X_processed[col] = X_processed[col] * X_processed[indicator]
        
        # Calculate average entertainment quality among offered options
        if present_ent_cols:
            # Create a mask of restaurants with at least one entertainment option
            has_entertainment = X_processed[[f'offers_{col.replace("Rating", "").strip().lower().replace(" ", "_")}' 
                            for col in present_ent_cols]].sum(axis=1) > 0
            
            # Calculate average rating only for restaurants with entertainment
            X_processed['avg_entertainment_quality'] = np.nan
            if has_entertainment.any():
                # Create a masked dataframe of ratings
                ratings = X_processed[present_ent_cols].copy()
                # Zero values would skew the average, so replace them with NaN
                for col in present_ent_cols:
                    indicator = f'offers_{col.replace("Rating", "").strip().lower().replace(" ", "_")}'
                    ratings.loc[X_processed[indicator] == 0, col] = np.nan
                
                # Calculate row-wise mean, ignoring NaN values
                X_processed.loc[has_entertainment, 'avg_entertainment_quality'] = ratings.mean(axis=1)
        
        return X_processed
    
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

            # Restaurant age interactions (maturity effect)
            ('restaurant_age_years', 'Overall Restaurant Rating'),
            ('restaurant_age_years', 'Food Rating'),
            ('restaurant_age_years', 'Hygiene Rating'),
        
            # Social media interactions
            ('Facebook Popularity Quotient', 'Instagram Popularity Quotient'),
        
            # Location-based interactions
            ('Restaurant Location_Near Party Hub', 'Resturant Tier'),
            ('Restaurant City Tier', 'Value for Money'),
        
            # Restaurant quality composite interactions
            ('Food Rating', 'Overall Restaurant Rating', 'Hygiene Rating'),
            ('Service', 'Staff Responsivness', 'Order Wait Time')
        ]
        
        # Create multiplicative interactions for features that exist
        for interaction in interactions:
            if len(interaction) == 2:
                col1, col2 = interaction
                if col1 in X.columns and col2 in X.columns:
                    X_with_interactions[f'{col1}_{col2}_interaction'] = X[col1] * X[col2]
            elif len(interaction) == 3:
                col1, col2, col3 = interaction
                if col1 in X.columns and col2 in X.columns and col3 in X.columns:
                    X_with_interactions[f'{col1}_{col2}_{col3}_composite'] = X[col1] * X[col2] * X[col3]
    
        
        # Create ratio features
        ratio_features = [
            ('Food Rating', 'Value for Money', 'premium_factor'),
            ('Order Wait Time', 'Staff Responsivness', 'service_efficiency'),
            ('Instagram Popularity Quotient', 'Facebook Popularity Quotient', 'social_media_balance'),
            ('Restaurant Zomato Rating', 'Overall Restaurant Rating', 'rating_discrepancy'),
            ('Food Rating', 'Hygiene Rating', 'quality_cleanliness_balance'),
            ('Lively', 'Comfortablility', 'energy_comfort_balance'),

            ('Food Rating', 'Staff Responsivness', 'food_service_balance'),
            ('Hygiene Rating', 'Staff Responsivness', 'hygiene_service_balance'),
            ('Restaurant Zomato Rating', 'Food Rating', 'external_internal_rating'),
            ('restaurant_age_years', 'Overall Restaurant Rating', 'rating_per_year')
        ]
        
        for col1, col2, name in ratio_features:
            if col1 in X.columns and col2 in X.columns:
                # Avoid division by 0
                X_with_interactions[name] = X[col1] / X[col2].replace(0, 0.001)

        # Create polynomial features for key ratings
        polynomial_features = [
            'Food Rating', 
            'Overall Restaurant Rating',
            'Restaurant Zomato Rating',
            'Hygiene Rating'
        ]
    
        for feature in polynomial_features:
            if feature in X.columns:
                X_with_interactions[f'{feature}_squared'] = X[feature] ** 2
        
        return X_with_interactions
    
    def _fit_select_features(self, X, y, k=40):
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

    def _create_enhanced_rating_features(self, X):
        """
        Create advanced rating-based features including composites, ratios, and non-linear transformations.
    
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame
            Data with enhanced rating features
        """
        self.logger.info('Creating enhanced rating features')
        X_enhanced = X.copy()

        # list of rating columns
        rating_cols = [col for col in X.columns if 'Rating' in col]

        # 1. weighted composite scores - prioritizing ratings with highest correlation to target
        if all(x in X.columns for x in ['Hygene Rating', 'Food Rating', 'Overall Restaurant Rating']):
            # primary quality score (weights based on EDA correlation analysis)
            X_enhanced['quality_score'] = (
                X['Hygene Rating'] * 0.5 +
                X['Food Rating'] * 0.3 +
                X['Overall Restaurant Rating'] * 0.2
            )

        if all(x in X.columns for x in ['Restaurant Zomato Rating', 'Food Rating']):
            # external vs internal perception ga
            X_enhanced['rating_perception_gap'] = X['Restaurant Zomato Rating'] - (X['Food Rating'] / 2.0)
            X_enhanced['rating_consistency'] = abs(X['Restaurant Zomato Rating'] - (X['Food Rating'] / 2.0))

        # 2. entertainment rating presence and quality
        entertainment_cols = ['Live Music Rating', 'Comedy Gigs Rating', 'Value Deals Rating', 'Live Sports Rating']
        present_cols = [f'has_{col.lower().replace(" ", "_")}' for col in entertainment_cols]

        # Count how many entertainment options the restaurant offers
        if all(x in X.columns for x in present_cols):
            X_enhanced['entertainment_options_count'] = X[present_cols].sum(axis=1)
    
        # Average entertainment quality (only for restaurants that offer entertainment)
        ent_cols_exist = [col for col in entertainment_cols if col in X.columns]
        if ent_cols_exist:
            # Create mask for restaurants with at least one entertainment option
            has_entertainment = (X[ent_cols_exist].notna().sum(axis=1) > 0)
            # Initialize with NaN
            X_enhanced['avg_entertainment_quality'] = float('nan')
            # Calculate only for those with entertainment
            X_enhanced.loc[has_entertainment, 'avg_entertainment_quality'] = (
                X.loc[has_entertainment, ent_cols_exist].mean(axis=1)
            )

        # 3. Non-linear transformations of key ratings
        for col in rating_cols:
            if col in X.columns:
                # Squared terms (already in your code, but expanded for more ratings)
                X_enhanced[f'{col}_squared'] = X[col] ** 2
            
                # Cubic terms for ratings with high correlation to target
                if col in ['Hygiene Rating', 'Food Rating']:
                    X_enhanced[f'{col}_cubed'] = X[col] ** 3
                
                # Exponential transformation to emphasize high ratings
                X_enhanced[f'{col}_exp'] = np.exp(X[col] / 10) - 1  # Scaled to avoid overflow

        # 4. Rating "variance" - restaurants with consistent vs inconsistent ratings
        if len(rating_cols) >= 3:
            present_rating_cols = [col for col in rating_cols if col in X.columns]
            # Normalize ratings to same scale (0-1) before calculating variance
            if len(present_rating_cols) >= 3:
                normalized_ratings = X[present_rating_cols].copy()
                for col in present_rating_cols:
                    max_val = 10 if 'Overall' in col or 'Food' in col or 'Hygiene' in col else 5
                    normalized_ratings[col] = X[col] / max_val
            
                X_enhanced['rating_variance'] = normalized_ratings.var(axis=1)
                X_enhanced['rating_range'] = normalized_ratings.max(axis=1) - normalized_ratings.min(axis=1)
    
        return X_enhanced

    def transform_target(self, y):
        """
        Transform the target variable using log transformation.

        Parameters:
        -----------
        y : pandas.Series
            Target variable
    
        Returns:
        --------
        pandas.Series
            Log-transformed target variable
        """
        self.logger.info("Applying log transformation to target variable")
    
        # Store original target mean and std for reference
        self.original_target_mean = y.mean()
        self.original_target_std = y.std()
    
        # Apply log transformation
        y_transformed = np.log1p(y)
    
        # Store transformation parameters
        self.target_transformed = True
    
        return y_transformed

    def inverse_transform_target(self, y_pred):
        """
        Inverse transform the predicted target variable.
    
        Parameters:
        -----------
        y_pred : numpy.ndarray or pandas.Series
            Log-transformed predictions
    
        Returns:
        --------
        numpy.ndarray
            Original scale predictions
        """
        self.logger.info("Inverting log transformation of target variable")
    
        # Apply expm1 to reverse log1p transformation
        return np.expm1(y_pred)