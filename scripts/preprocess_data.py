import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

'''
function to modify data such that the resulting data only contains white win samples
'''
def drop_red_wine_samples(df):
    return df.iloc[1599:]

'''
function to perform data splitting and save the resulting files in the given path
'''
def split_data(df, path_to_processed_data):
    # split target with other features
    X = df.drop(columns=['quality'])
    y = df['quality']

    # define train, validation, and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=123)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.21, shuffle=True, random_state=123)
    
    # save train data
    X_train.to_csv(f'{path_to_processed_data}/train_data/X_train.csv', index=False)
    y_train.to_csv(f'{path_to_processed_data}/train_data/y_train.csv', index=False)
    
    # save train data
    X_valid.to_csv(f'{path_to_processed_data}/valid_data/X_valid.csv', index=False)
    y_valid.to_csv(f'{path_to_processed_data}/valid_data/y_valid.csv', index=False)

    # save test data
    X_test.to_csv(f'{path_to_processed_data}/test_data/X_test.csv', index=False)
    y_test.to_csv(f'{path_to_processed_data}/test_data/y_test.csv', index=False)
    
    print(f"Processed data saved in {path_to_processed_data}")
    
if __name__ == "__main__":
    # define parser and add args
    parser = argparse.ArgumentParser(description='Clean and split data')
    parser.add_argument('--path_to_raw_data', type=str, help='path to load raw data')
    parser.add_argument('--path_to_processed_data', type=str, help='path to save processed data')
    
    # parse args
    args = parser.parse_args()
    
    # load raw data saved
    df = pd.read_csv(args.path_to_raw_data)
    
    # clean data by dropping red wine samples
    clean_df = drop_red_wine_samples(df)
    
    # split data into train and test set
    split_data(clean_df, args.path_to_processed_data)
    