# src/data/preprocess.py

import pandas as pd
import os

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