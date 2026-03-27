from src.eda import (
    save_feature_description,
    save_wine_quality_hist,
    save_feature_dist
)

import argparse
import pandas as pd
import matplotlib.pyplot as plt


'''
function to save pie chart that presents data split ratio of train, validation, and test data
'''
def save_data_split_pie_chart(X_train, X_valid, X_test, path_to_save):
    # compute size for each dataset
    data_size = len(X_train) + len(X_valid) + len(X_test)
    train_set_size = len(X_train) / data_size
    valid_set_size = len(X_valid) / data_size
    test_set_size = len(X_test) / data_size

    # define pie chart
    plt.figure(figsize=(7, 7))
    pie_colors = ['seagreen', 'gold', 'tomato']
    plt.pie(
        [train_set_size, valid_set_size, test_set_size],
        labels=['Train Set', 'Validation Set', 'Test Set'],
        colors=pie_colors,
        autopct='%1.0f%%'
    )
    plt.title('Data Split Proportion')

    # save plot
    plt.savefig(path_to_save)

    print(f'Data split pie chart saved in {path_to_save}')


'''
function to save correlation matrix
'''
def save_corr_mat(X_train, y_train, path_to_save):
    # define corr mat
    corr = pd.concat([X_train, y_train], axis=1).corr()

    plt.figure(figsize=(7, 7))
    plt.imshow(corr, cmap="coolwarm", interpolation="none")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix")
    plt.tight_layout()

    # save plot
    plt.savefig(path_to_save)

    print(f'Correlation Matrix saved in {path_to_save}')


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