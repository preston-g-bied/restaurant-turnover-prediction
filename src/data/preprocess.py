# src/data/preprocess.py

import pandas as pd
import os

def handle_outliers(df, columns, method='winsorize', threshold=1.5):
    """
    Handle outliers in specified columns using various methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with potential outliers
    columns : list or str
        Column name(s) to check for outliers
    method : str
        Method to handle outliers ('winsorize', 'remove', or 'log')
    threshold : float
        IQR multiplier for defining outliers (default: 1.5)
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with handled outliers
    """
    if isinstance(columns, str):
        columns = [columns]
        
    df_clean = df.copy()
    
    for column in columns:
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in dataframe")
            continue
            
        # Calculate IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Count outliers for reporting
        outliers_count = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
        outlier_percentage = (outliers_count / df.shape[0]) * 100
        print(f"Found {outliers_count} outliers ({outlier_percentage:.2f}%) in '{column}'")
        
        if method == 'winsorize':
            # Cap values at the boundaries
            df_clean[column] = df_clean[column].clip(lower_bound, upper_bound)
            print(f"Winsorized outliers in '{column}' to range [{lower_bound:.2f}, {upper_bound:.2f}]")
            
        elif method == 'remove':
            # Remove rows with outliers
            df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
            print(f"Removed {outliers_count} rows with outliers in '{column}'")
            
        elif method == 'log':
            # Apply log transformation if not already done
            # This is handled by your existing transform_numeric_features function
            pass
            
    return df_clean

def load_data(data_path):
    """
    Load the raw data from CSV files.
    
    Parameters:
    -----------
    data_path : str
        Path to the raw data directory
    
    Returns:
    --------
    pandas.DataFrame, pandas.DataFrame
        Training and test dataframes
    """
    train_path = os.path.join(data_path, 'train.csv')
    test_path = os.path.join(data_path, 'test.csv')

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print(f'Loaded training data: {train_data.shape}')
    print(f'Loaded test data: {test_data.shape}')

    return train_data, test_data

def save_processed_data(X_train, y_train, X_test, test_ids, output_path):
    """
    Save the processed data to CSV files.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Processed training features
    y_train : pandas.Series
        Target variable
    X_test : pandas.DataFrame
        Processed test features
    test_ids : pandas.Series
        Test registration numbers
    output_path : str
        Path to save processed data
    """
    # create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # save training data with features and target
    train_processed = pd.concat([X_train, y_train], axis=1)
    train_processed.to_csv(os.path.join(output_path, 'train_processed.csv'), index=False)
    
    # save test data with registration numbers
    test_processed = pd.concat([test_ids.reset_index(drop=True), X_test], axis=1)
    test_processed.to_csv(os.path.join(output_path, 'test_processed.csv'), index=False)

    # save feature list for reference
    pd.Series(X_train.columns.tolist()).to_csv(
        os.path.join(output_path, 'selected_features.csv'), index=False, header=['feature']
    )

    print(f"Saved processed training data: {train_processed.shape}")
    print(f"Saved processed test data: {test_processed.shape}")