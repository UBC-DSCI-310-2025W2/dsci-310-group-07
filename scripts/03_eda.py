import argparse
import pandas as pd

from dsci310_2025w2_winequalitypy.eda_utils import (
    save_feature_description,
    save_data_split_pie_chart,
    save_wine_quality_hist,
    save_feature_dist,
    save_corr_mat
)

if __name__ == "__main__":
    # define parser and add args
    parser = argparse.ArgumentParser(
        description='Save tables and figures exploratory data visualizations and tables'
    )
    parser.add_argument(
        '--path_to_processed_data',
        type=str,
        help='Path to train, validation, and test data'
    )
    parser.add_argument(
        '--path_to_feature_description',
        type=str,
        help='Path to save feature description table'
    )
    parser.add_argument(
        '--path_to_pie_chart',
        type=str,
        help='Path to save data split pie chart'
    )
    parser.add_argument(
        '--path_to_wine_quality_hist',
        type=str,
        help='Path to save wine quality histogram'
    )
    parser.add_argument(
        '--path_to_feature_dist',
        type=str,
        help='Path to save feature distributions'
    )
    parser.add_argument(
        '--path_to_corr_mat',
        type=str,
        help='Path to save correlation matrix'
    )

    # parse args
    args = parser.parse_args()

    # load X_train, X_valid, X_test, y_train
    X_train = pd.read_csv(
        f'{args.path_to_processed_data}/train_data/X_train.csv'
    )
    y_train = pd.read_csv(
        f'{args.path_to_processed_data}/train_data/y_train.csv'
    )

    X_valid = pd.read_csv(
        f'{args.path_to_processed_data}/valid_data/X_valid.csv'
    )
    X_test = pd.read_csv(
        f'{args.path_to_processed_data}/test_data/X_test.csv'
    )

    # save plots and tables
    save_feature_description(args.path_to_feature_description)
    save_data_split_pie_chart(
        X_train,
        X_valid,
        X_test,
        args.path_to_pie_chart
    )
    save_wine_quality_hist(
        y_train,
        args.path_to_wine_quality_hist
    )
    save_feature_dist(
        X_train,
        args.path_to_feature_dist
    )
    save_corr_mat(
        X_train,
        y_train,
        args.path_to_corr_mat
    )