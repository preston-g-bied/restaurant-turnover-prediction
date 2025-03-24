# src/features/build_features.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression
import category_encoders as ce
from datetime import datetime

# Complete city name mapping to standardize all variations
city_mapping = {
    # New variations found in test data
    'GURAGAON': 'gurgaon',
    'new dehli': 'new delhi',
    'Ambala City': 'ambala',
    ' bangalore': 'bangalore',
    'Mohali': 'chandigarh',
    'RAE BARELI': 'rae bareli',
    'Phagwara': 'phagwara',
    'latur (Maharashtra )': 'latur',
    'sampla': 'sampla',
    'jAipur': 'jaipur',
    'BANGLORE': 'bangalore',
    'Haldia': 'haldia',
    'Mainpuri': 'mainpuri',
    'BANGALORE ': 'bangalore',
    'ranchi': 'ranchi',
    'karnal': 'karnal',
    'Karad': 'karad',
    'haryana': 'unknown',
    'Vellore': 'vellore',
    'Patiala': 'patiala',
    'kanpur': 'kanpur',
    'Yamuna Nagar': 'yamuna nagar',
    'Rourkela': 'rourkela',
    'Banaglore': 'bangalore',
    'Baripada': 'baripada',
    'Dausa': 'dausa',
    'Asifabadbanglore': 'bangalore',
    'Rajpura': 'rajpura',
    'Dammam': 'international',
    'Pilani': 'pilani',
    'shahibabad': 'ghaziabad',
    'MEERUT': 'meerut',
    'Guwahati': 'guwahati',
    'sambalpur': 'sambalpur',
    
    # Bangalore variations
    'Bangalore': 'bangalore',
    'BANGALORE': 'bangalore',
    'bangalore': 'bangalore',
    'Banglore': 'bangalore',
    'Banagalore': 'bangalore',
    'BAngalore': 'bangalore',
    'Bangalore ': 'bangalore',
    'Bengaluru': 'bangalore',
    'banglore': 'bangalore',
    
    # Mumbai variations
    'Mumbai': 'mumbai',
    'MUMBAI': 'mumbai',
    'mumbai': 'mumbai',
    'mumbai ': 'mumbai',
    ' mumbai': 'mumbai',
    'Navi Mumbai': 'navi mumbai',
    'NAVI MUMBAI': 'navi mumbai',
    'Navi mumbai': 'navi mumbai',
    
    # Delhi variations
    'Delhi': 'delhi',
    'DELHI': 'delhi',
    'delhi': 'delhi',
    ' Delhi': 'delhi',
    'New Delhi': 'new delhi',
    'New delhi': 'new delhi',
    'new delhi': 'new delhi',
    'NEW DELHI': 'new delhi',
    'New Delhi ': 'new delhi',
    'Delhi/NCR': 'delhi',
    
    # Chennai variations
    'Chennai': 'chennai',
    'CHENNAI': 'chennai',
    'chennai': 'chennai',
    'Chennai ': 'chennai',
    ' Chennai': 'chennai',
    'chennai ': 'chennai',
    
    # Hyderabad variations
    'Hyderabad': 'hyderabad',
    'HYDERABAD': 'hyderabad',
    'hyderabad': 'hyderabad',
    'Hyderabad ': 'hyderabad',
    'hyderabad ': 'hyderabad',
    'hderabad': 'hyderabad',
    'hyderabad(bhadurpally)': 'hyderabad',
    'Secunderabad': 'hyderabad',
    
    # Pune variations
    'Pune': 'pune',
    'PUNE': 'pune',
    'pune': 'pune',
    'Pune ': 'pune',
    'pune ': 'pune',
    ' Pune': 'pune',
    'punr': 'pune',
    
    # Noida variations
    'Noida': 'noida',
    'NOIDA': 'noida',
    'noida': 'noida',
    'noida ': 'noida',
    'Noida ': 'noida',
    'Nouda': 'noida',
    
    # Gurgaon variations
    'Gurgaon': 'gurgaon',
    'GURGAON': 'gurgaon',
    'gurgaon': 'gurgaon',
    'Gurgaon ': 'gurgaon',
    'GURGOAN': 'gurgaon',
    'Gurgoan': 'gurgaon',
    'gurgoan': 'gurgaon',
    'Gurga': 'gurgaon',
    
    # Kolkata variations
    'Kolkata': 'kolkata',
    'KOLKATA': 'kolkata',
    'kolkata': 'kolkata',
    'Kolkata`': 'kolkata',
    'Kolkata ': 'kolkata',
    
    # Ahmedabad variations
    'Ahmedabad': 'ahmedabad',
    'ahmedabad': 'ahmedabad',
    'Ahmedabad ': 'ahmedabad',
    
    # Jaipur variations
    'Jaipur': 'jaipur',
    'jaipur': 'jaipur',
    'Jaipur ': 'jaipur',
    
    # Indore variations
    'Indore': 'indore',
    'indore': 'indore',
    
    # Lucknow variations
    'Lucknow': 'lucknow',
    'LUCKNOW': 'lucknow',
    'lucknow': 'lucknow',
    'Lucknow ': 'lucknow',
    
    # Bhubaneswar variations
    'Bhubaneswar': 'bhubaneswar',
    'Bhubaneshwar': 'bhubaneswar',
    'Bhubneshwar': 'bhubaneswar',
    'bhubaneswar': 'bhubaneswar',
    'BHUBANESWAR': 'bhubaneswar',
    'Bhubaneswar ': 'bhubaneswar',
    
    # Greater Noida variations
    'Greater Noida': 'greater noida',
    'Greater noida': 'greater noida',
    'GREATER NOIDA': 'greater noida',
    'Greater NOIDA': 'greater noida',
    
    # Trivandrum/Thiruvananthapuram variations
    'Trivandrum': 'trivandrum',
    'TRIVANDRUM': 'trivandrum',
    'trivandrum': 'trivandrum',
    'Trivandrum ': 'trivandrum',
    'Thiruvananthapuram': 'trivandrum',
    'Technopark, Trivandrum': 'trivandrum',
    
    # Mysore variations
    'Mysore': 'mysore',
    'mysore': 'mysore',
    'Mysore ': 'mysore',
    
    # Kochi/Cochin variations
    'Kochi': 'kochi',
    'Kochi/Cochin': 'kochi',
    'Ernakulam': 'kochi',
    'Kochi/Cochin, Chennai and Coimbatore': 'multiple',
    
    # Vadodara/Baroda variations
    'Vadodara': 'vadodara',
    'Baroda': 'vadodara',
    
    # Visakhapatnam/Vizag variations
    'Visakhapatnam': 'visakhapatnam',
    'Vizag': 'visakhapatnam',
    'VIZAG': 'visakhapatnam',
    'vizag': 'visakhapatnam',
    'vsakhapttnam': 'visakhapatnam',
    
    # Chandigarh variations
    'Chandigarh': 'chandigarh',
    'chandigarh': 'chandigarh',
    'Chandigarh ': 'chandigarh',
    'mohali': 'chandigarh',
    'Punchkula': 'chandigarh',
    'Panchkula': 'chandigarh',
    'Panchkula ': 'chandigarh',
    
    # Patna variations
    'Patna': 'patna',
    'PATNA': 'patna',
    'patna': 'patna',
    
    # Ghaziabad variations
    'Ghaziabad': 'ghaziabad',
    'ghaziabad': 'ghaziabad',
    'Gaziabaad': 'ghaziabad',
    'Gajiabaad': 'ghaziabad',
    'Sahibabad': 'ghaziabad',
    'Indirapuram, Ghaziabad': 'ghaziabad',
    
    # Nagpur variations
    'Nagpur': 'nagpur',
    'Nagpur ': 'nagpur',
    
    # Kanpur variations
    'Kanpur': 'kanpur',
    'KANPUR': 'kanpur',
    'Kanpur ': 'kanpur',
    
    # Coimbatore variations
    'Coimbatore': 'coimbatore',
    'coimbatore': 'coimbatore',
    
    # Thane variations
    'Thane': 'thane',
    'THANE': 'thane',
    'thane': 'thane',
    
    # Bhopal variations
    'Bhopal': 'bhopal',
    'BHOPAL': 'bhopal',
    'Bhopal ': 'bhopal',
    
    # Manesar variations
    'Manesar': 'manesar',
    'manesar': 'manesar',
    
    # Ranchi variations
    'Ranchi': 'ranchi',
    'Ranchi ': 'ranchi',
    
    # Madurai variations
    'Madurai': 'madurai',
    'Madurai ': 'madurai',
    
    # Kota variations
    'Kota': 'kota',
    'KOTA': 'kota',
    
    # Muzaffarpur/Muzzafarpur variations
    'Muzaffarpur': 'muzaffarpur',
    'muzzafarpur': 'muzaffarpur',
    
    # Pondicherry variations
    'Pondicherry': 'pondicherry',
    'pondy': 'pondicherry',
    'pondi': 'pondicherry',
    
    # Faridabad variations
    'Faridabad': 'faridabad',
    
    # Udaipur variations
    'Udaipur': 'udaipur',
    'udaipur': 'udaipur',
    
    # Jamshedpur variations
    'Jamshedpur': 'jamshedpur',
    
    # Dehradun variations
    'Dehradun': 'dehradun',
    'dehradun': 'dehradun',
    
    # Meerut variations
    'Meerut': 'meerut',
    'meerut': 'meerut',
    
    # International locations
    'Dubai': 'international',
    'LONDON': 'international',
    'Australia': 'international',
    'RAS AL KHAIMAH': 'international',
    
    # Special cases
    '-1': 'unknown_city',
    'india': 'unknown_city',
    'ncr': 'delhi',
    'orissa': 'bhubaneswar',
    'Rajasthan': 'unknown_city',
    'keral': 'kerala',
    'bihar': 'patna',
    
    # Additional cities with no variations
    'Jhansi': 'jhansi',
    'Mangalore': 'mangalore',
    'Rewari': 'rewari',
    'Bhiwadi': 'bhiwadi',
    'Rajkot': 'rajkot',
    'Jodhpur': 'jodhpur',
    'Gandhi Nagar': 'gandhinagar',
    'Gandhinagar': 'gandhinagar',
    'Gandhinagar ': 'gandhinagar',
    'Una': 'una',
    'Daman and Diu': 'daman and diu',
    'Bhagalpur': 'bhagalpur',
    'Bankura': 'bankura',
    'Vijayawada': 'vijayawada',
    'Beawar': 'beawar',
    'Alwar': 'alwar',
    'Siliguri': 'siliguri',
    'Siliguri ': 'siliguri',
    'raipur': 'raipur',
    'Raipur': 'raipur',
    'Bulandshahar': 'bulandshahar',
    'Haridwar': 'haridwar',
    'Raigarh': 'raigarh',
    'Jabalpur': 'jabalpur',
    'Unnao': 'unnao',
    'Aurangabad': 'aurangabad',
    'Belgaum': 'belgaum',
    'Rudrapur': 'rudrapur',
    'Dharamshala': 'dharamshala',
    'Hissar': 'hisar',
    'sonepat': 'sonepat',
    'Sonipat': 'sonepat',
    'Pantnagar': 'pantnagar',
    'Jagdalpur': 'jagdalpur',
    'angul': 'angul',
    ' ariyalur': 'ariyalur',
    'Jowai': 'jowai',
    'Neemrana': 'neemrana',
    'Tirupathi': 'tirupati',
    'Tirupati': 'tirupati',
    'Calicut': 'calicut',
    'Ahmednagar': 'ahmednagar',
    'Nashik': 'nashik',
    'Nasikcity': 'nashik',
    'Bellary': 'bellary',
    'Ludhiana': 'ludhiana',
    'Muzaffarnagar': 'muzaffarnagar',
    'Gagret': 'gagret',
    'Gwalior': 'gwalior',
    'Bareli': 'bareilly',
    'Hospete': 'hospet',
    'Miryalaguda': 'miryalaguda',
    'Dharuhera': 'dharuhera',
    'Ganjam': 'ganjam',
    'Hubli': 'hubli',
    'Agra': 'agra',
    'Trichy': 'trichy',
    'kudankulam ,tarapur': 'kudankulam',
    'Ongole': 'ongole',
    'Sambalpur': 'sambalpur',
    'Bundi': 'bundi',
    'SADULPUR,RAJGARH,DISTT-CHURU,RAJASTHAN': 'sadulpur',
    'AM': 'unknown',
    'Bikaner': 'bikaner',
    'Asansol': 'asansol',
    'Tirunelvelli': 'tirunelveli',
    'Bilaspur': 'bilaspur',
    'Chandrapur': 'chandrapur',
    'Nanded': 'nanded',
    'Dharmapuri': 'dharmapuri',
    'Vandavasi': 'vandavasi',
    'Rohtak': 'rohtak',
    'Salem': 'salem',
    'Bharuch': 'bharuch',
    'Tornagallu': 'tornagallu',
    'Jaspur': 'jaspur',
    'Burdwan': 'burdwan',
    'Shimla': 'shimla',
    'Jammu': 'jammu',
    'Shahdol': 'shahdol',
    'SHAHDOL': 'shahdol',
    'Muvattupuzha': 'muvattupuzha',
    'Ratnagiri': 'ratnagiri',
    'Jhajjar': 'jhajjar',
    'Gulbarga': 'gulbarga',
    'Nalagarh': 'nalagarh',
    'Jamnagar': 'jamnagar',
    'jamnagar': 'jamnagar',
    'Gonda': 'gonda',
    'kharagpur': 'kharagpur',
    'Joshimath': 'joshimath',
    'Bathinda': 'bathinda',
    'kala amb ': 'kala amb',
    'Karnal': 'karnal',
    'Baddi HP': 'baddi',
    'Nagari': 'nagari',
    'Mettur, Tamil Nadu ': 'mettur',
    'Durgapur': 'durgapur',
    'Surat': 'surat',
    'Kurnool': 'kurnool',
    'kolhapur': 'kolhapur',
    'Bhilai': 'bhilai',
    'Bahadurgarh': 'bahadurgarh',
    'Rayagada, Odisha': 'rayagada',
    'kakinada': 'kakinada',
    'Varanasi': 'varanasi',
    'Nellore': 'nellore',
    'Howrah': 'howrah',
    'Trichur': 'thrissur',
    'Ambala': 'ambala',
    'Khopoli': 'khopoli',
    'Roorkee': 'roorkee',
    'Allahabad': 'prayagraj',
    'Jalandhar': 'jalandhar',
    'vapi': 'vapi',
    'PILANI': 'pilani',
    'singaruli': 'singrauli',
    'CHEYYAR': 'cheyyar'
}

def handle_missing_values(df, strategy='median'):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with missing values
    strategy : str
        Imputation strategy ('mean', 'median', 'knn')
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with imputed values and missing indicators
    """
    df_imputed = df.copy()

    # create missing indicators for features with high missingness (>20%)
    high_missingness = ['Live Sports Rating', 'Value Deals Rating', 
                        'Comedy Gigs Rating', 'Live Music Rating']
    
    for col in high_missingness:
        if col in df.columns:
            df_imputed[f'has_{col.lower().replace(" ", "_")}'] = df[col].notna().astype(int)

    # simple implementation for less missing numeric features
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns

    if strategy in ['mean', 'median']:
        imputer = SimpleImputer(strategy=strategy)
        df_imputed[numeric_features] = imputer.fit_transform(df[numeric_features])
    elif strategy == 'knn':
        # KNN imputation for better accuracy
        imputer = KNNImputer(n_neighbors=5)
        df_imputed[numeric_features] = imputer.fit_transform(df[numeric_features])

    return df_imputed

def extract_temporal_features(df, date_col='Opening Day of Restaurant'):
    """
    Extract temporal features from date column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with date column
    date_col : str
        Name of the date column
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with additional temporal features
    """
    df_with_temporal = df.copy()

    # convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df_with_temporal[date_col] = pd.to_datetime(df[date_col],
                                                    format='%d-%m-%Y',
                                                    errors='coerce')
    
    # current date for age calculation
    current_date = datetime.now()

    # create temporal features
    df_with_temporal['restaurant_age_days'] = (current_date - df_with_temporal[date_col]).dt.days
    df_with_temporal['restaurant_age_years'] = df_with_temporal['restaurant_age_days'] / 365.25
    df_with_temporal['opening_year'] = df_with_temporal[date_col].dt.year
    df_with_temporal['opening_month'] = df_with_temporal[date_col].dt.month
    df_with_temporal['opening_day'] = df_with_temporal[date_col].dt.day
    df_with_temporal['opening_day_of_week'] = df_with_temporal[date_col].dt.dayofweek
    df_with_temporal['opening_quarter'] = df_with_temporal[date_col].dt.quarter

    # create seasonal indicators
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    df_with_temporal['opening_season'] = df_with_temporal['opening_month'].map(season_map)

    # drop original date column
    df_with_temporal = df_with_temporal.drop(columns=[date_col])

    return df_with_temporal

def encode_categorical_features(df, target_col=None, method='auto', cv_folds=5):
    """
    Encode categorical features with special handling for placeholder values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with categorical features
    target_col : str
        Target column name (for target encoding)
    method : str
        Encoding method ('auto', 'onehot', 'label', 'target')
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with encoded categorical features
    """
    df_encoded = df.copy()

    # identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # handle placeholder values before encoding
    if 'City' in df_encoded.columns:
        # apply the mapping after lowercase conversion
        df_encoded['City'] = df_encoded['City'].str.lower().map(
            lambda x: city_mapping.get(x, x)
        )

    for col in categorical_cols:
        # standardize text-based features
        if col in ['Cuisine', 'Restaurant Theme']:
            df_encoded[col] = df_encoded[col].str.lower().str.strip()

        # count unique values for each categorical features
        n_unique = df_encoded[col].nunique()

        # choose encoding method based on cardinality or specified method
        if method == 'auto':
            if n_unique <= 2:   # binary features
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])

            elif n_unique <= 15:    # low cardinality features
                # one-hot encoding
                ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                encoded = ohe.fit_transform(df_encoded[[col]])

                # create column names
                encoded_cols = [f'{col}_{cat}' for cat in ohe.categories_[0][1:]]

                # add encoded columns to dataframe
                for i, encoded_col in enumerate(encoded_cols):
                    df_encoded[encoded_col] = encoded[:, i]

                # drop original column
                df_encoded.drop(col, axis=1, inplace=True)

            else:   # high cardinality features
                if target_col is not None:
                    from sklearn.model_selection import KFold
                    
                    # Create a temporary column for the encoded values
                    encoded_col_name = f'{col}_target_encoded'
                    df_encoded[encoded_col_name] = np.nan
                    
                    # Create CV folds
                    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    
                    # If we have the target column in the dataframe
                    if target_col in df_encoded.columns:
                        # For each fold
                        for train_idx, val_idx in kf.split(df_encoded):
                            # Get training data for this fold
                            X_train = df_encoded.iloc[train_idx]
                            
                            # Calculate target mean for each category in training data
                            means = X_train.groupby(col)[target_col].mean()
                            
                            # Map means to validation data
                            for cat, mean in means.items():
                                mask = (df_encoded.iloc[val_idx][col] == cat)
                                df_encoded.loc[df_encoded.index[val_idx][mask], encoded_col_name] = mean
                            
                            # Fill NaNs with global mean
                            global_mean = X_train[target_col].mean()
                            df_encoded.loc[df_encoded.index[val_idx], encoded_col_name] = \
                                df_encoded.loc[df_encoded.index[val_idx], encoded_col_name].fillna(global_mean)
                        
                        # Drop original column
                        df_encoded.drop(col, axis=1, inplace=True)
                    else:
                        # For test data without target column, use label encoding as fallback
                        le = LabelEncoder()
                        df_encoded[col] = le.fit_transform(df_encoded[col])
                else:
                    # label encoding as fallback
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col])

        elif method == 'onehot':
            # one-hot encoding for all categorical features
            ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            encoded = ohe.fit_transform(df_encoded[[col]])

            # create column names
            encoded_cols = [f'{col}_{cat}' for cat in ohe.categories_[0][1:]]

            # add encoded columns to dataframe
            for i, encoded_col in enumerate(encoded_cols):
                df_encoded[encoded_col] = encoded[:, i]

            # drop original column
            df_encoded.drop(col, axis=1, inplace=True)

        elif method == 'label':
            # label encoding for all categorical features
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])

        elif method == 'target' and target_col is not None:
            # target encoding for all categorical features
            te = ce.TargetEncoder()
            df_encoded[col] = te.fit_transform(df_encoded[col], df_encoded[target_col])

    return df_encoded

def transform_numeric_features(df, log_transform=None, scaling=None):
    """
    Transform numeric features (log transformation, scaling).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with numeric features
    log_transform : list
        List of columns to apply log transformation
    scaling : str
        Scaling method ('standard', 'minmax')
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with transformed numeric features
    """
    df_transformed = df.copy()

    # log transformation
    if log_transform is not None:
        for col in log_transform:
            if col in df.columns:
                # add a small constant to avoid log(0)
                df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])

    # scaling
    if scaling == 'standard':
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        df_transformed[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df_transformed

def create_interaction_features(df, interactions=None):
    """
    Create interaction features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    interactions : list of tuples
        List of feature pairs to create interactions
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with additional interaction features
    """
    df_with_interactions = df.copy()

    if interactions is None:
        # default interactions based on domain knowledge
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
            
            # Location-based interactions
            ('Restaurant Location_Near Party Hub', 'Resturant Tier'),
            ('Restaurant City Tier', 'Restaurant Type_Bar')
        ]

    # create multiplicative interactions
    for col1, col2 in interactions:
        if col1 in df.columns and col2 in df.columns:
            df_with_interactions[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]

    # create ratio features
    ratio_features = [
        ('Food Rating', 'Value for Money', 'premium_factor'),
        ('Order Wait Time', 'Staff Responsivness', 'service_efficiency'),
        ('Instagram Popularity Quotient', 'Facebook Popularity Quotient', 'social_media_balance'),
        ('Restaurant Zomato Rating', 'Overall Restaurant Rating', 'rating_discrepancy'),
        ('Food Rating', 'Hygiene Rating', 'quality_cleanliness_balance'),
        ('Lively', 'Comfortablility', 'energy_comfort_balance')
    ]

    for col1, col2, name in ratio_features:
        if col1 in df.columns and col2 in df.columns:
            # avoid division by 0
            df_with_interactions[name] = df[col1] / df[col2].replace(0, 0.001)

    return df_with_interactions

def select_features(X, y, k=20):
    """
    Select top k features based on univariate statistical tests.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature dataframe
    y : pandas.Series
        Target variable
    k : int
        Number of features to select
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with selected features
    list
        List of selected feature names
    """
    # create selector
    selector = SelectKBest(f_regression, k=k)

    # fit and transform
    X_selected = selector.fit_transform(X, y)

    # get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices].tolist()

    # return selected features dataframe and feature names
    return pd.DataFrame(X_selected, columns=selected_features), selected_features