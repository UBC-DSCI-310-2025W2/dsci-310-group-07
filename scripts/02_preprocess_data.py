import argparse
import pandas as pd

from src.preprocess_utils import drop_red_wine_samples, split_data

if __name__ == "__main__":
    # define parser and add args
    parser = argparse.ArgumentParser(description="Clean and split data")
    parser.add_argument("--path_to_raw_data", type=str, help="path to load raw data")
    parser.add_argument(
        "--path_to_processed_data", type=str, help="path to save processed data"
    )
    
    # parse args
    args = parser.parse_args()
    
    # load raw data saved
    df = pd.read_csv(args.path_to_raw_data)
    
    # clean data by dropping red wine samples
    clean_df = drop_red_wine_samples(df)
    
    # split data into train, validation, and test set
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(clean_df)
    
    # save each dataset to its corrsponding csv file
    X_train.to_csv(f"{args.path_to_processed_data}/train_data/X_train.csv", index=False)
    y_train.to_csv(f"{args.path_to_processed_data}/train_data/y_train.csv", index=False)

    X_valid.to_csv(f"{args.path_to_processed_data}/valid_data/X_valid.csv", index=False)
    y_valid.to_csv(f"{args.path_to_processed_data}/valid_data/y_valid.csv", index=False)

    X_test.to_csv(f"{args.path_to_processed_data}/test_data/X_test.csv", index=False)
    y_test.to_csv(f"{args.path_to_processed_data}/test_data/y_test.csv", index=False)

    print(f"Processed data saved in {args.path_to_processed_data}")