import os
import sys
import argparse
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from dsci310_2025w2_winequalitypy.train_and_evaluate_utils import (
    save_best_params, 
    save_test_pred, 
    save_conf_mat
)

if __name__ == "__main__":
    # define parser and add args
    parser = argparse.ArgumentParser(
        description='Train and evaluate model and save evaluation results'
    )
    parser.add_argument(
        '--path_to_processed_data', 
        type=str, 
        help='Path to train, validation, and test data'
    )
    parser.add_argument(
        '--path_to_prediction_data', 
        type=str, 
        help='Path to save test prediction data'
    )
    parser.add_argument(
        '--path_to_model_params', 
        type=str, 
        help='Path to save best model parameters'
    )
    parser.add_argument(
        '--path_to_conf_mat', 
        type=str, 
        help='Path to save confusion matrix on test data'
    )
    
    # parse args
    args = parser.parse_args()
    
    # load data
    X_train = pd.read_csv(
        f'{args.path_to_processed_data}/train_data/X_train.csv'
    )
    y_train = pd.read_csv(
        f'{args.path_to_processed_data}/train_data/y_train.csv'
    )
    X_valid = pd.read_csv(
        f'{args.path_to_processed_data}/valid_data/X_valid.csv'
    )
    y_valid = pd.read_csv(
        f'{args.path_to_processed_data}/valid_data/y_valid.csv'
    )
    X_test = pd.read_csv(
        f'{args.path_to_processed_data}/test_data/X_test.csv'
    )
    y_test = pd.read_csv(
        f'{args.path_to_processed_data}/test_data/y_test.csv'
    )
    
    # define random forest model
    rf_model = RandomForestClassifier(random_state=123)
    
    # define param grid for grid search
    param_grid = {
        'n_estimators': [10, 50, 100, 200, 400],
        'max_depth': [2, 10, 20, 30, 40, 50]
    }
    
    # define and fit grid search
    grid_search = GridSearchCV(
        rf_model, 
        param_grid, 
        cv=2, 
        return_train_score=True, 
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train.values.ravel())

    # save best params
    save_best_params(grid_search, args.path_to_model_params)
    
    # retrieve best performing model
    best_model = grid_search.best_estimator_
    
    # predict on test data
    pred = best_model.predict(X_test)
    
    # save test predictions
    save_test_pred(pred, args.path_to_prediction_data)
    
    # save confusion matrix
    save_conf_mat(y_test, pred, args.path_to_conf_mat)