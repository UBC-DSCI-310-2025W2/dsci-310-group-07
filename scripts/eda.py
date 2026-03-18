import argparse
import pandas as pd
import matplotlib.pyplot as plt

'''
function to save feature descripition table in the specified path
'''
def save_feature_description(path_to_output):
    # define table content
    table_data = [
        ["fixed_acidity", "Feature", "Continuous", "g/dm³", "Fixed acids (primarily tartaric acid) that do not evaporate easily.", "No"],
        ["volatile_acidity", "Feature", "Continuous", "g/dm³", "Amount of acetic acid in wine; high levels can lead to an unpleasant vinegar taste.", "No"],
        ["citric_acid", "Feature", "Continuous", "g/dm³", "Citric acid content, which can add freshness and flavor to wine.", "No"],
        ["residual_sugar", "Feature", "Continuous", "g/dm³", "Amount of sugar remaining after fermentation stops.", "No"],
        ["chlorides", "Feature", "Continuous", "g/dm³", "Salt content in the wine.", "No"],
        ["free_sulfur_dioxide", "Feature", "Continuous", "mg/dm³", "Free form of sulfur dioxide that prevents microbial growth and oxidation.", "No"],
        ["total_sulfur_dioxide", "Feature", "Continuous", "mg/dm³", "Total amount of sulfur dioxide (free + bound forms).", "No"],
        ["density", "Feature", "Continuous", "g/cm³", "Density of the wine, influenced by alcohol and sugar content.", "No"],
        ["pH", "Feature", "Continuous", "–", "Acidity level of the wine; lower values indicate higher acidity.", "No"],
        ["sulphates", "Feature", "Continuous", "g/dm³", "Potassium sulphate concentration, contributing to sulfur dioxide levels and preservation.", "No"],
        ["alcohol", "Feature", "Continuous", "% (vol)", "Alcohol content of the wine.", "No"],
        ["quality", "Target", "Integer", "Score (0–10)", "Wine quality score based on expert sensory evaluation.", "No"]
    ]
    
    # define features
    feature_names = ["Feature Name ", "Role", "Type", "Units", "Description", "Missing Values"]
    
    # save df
    df = pd.DataFrame(table_data, columns=feature_names)
    path_to_save = f'{path_to_output}/feature_description.csv'
    df.to_csv(path_to_save)
    
    print(f'feature description table saved in {path_to_save}')
    
    
'''
function to save pie chart that presents data split ratio of train, validation, and test data
'''
def save_data_split_pie_chart(X_train, X_valid, X_test, path_to_output):
    # compute size for each dataset
    data_size = len(X_train) + len(X_valid) + len(X_test)
    train_set_size = len(X_train) / data_size
    valid_set_size = len(X_valid) / data_size
    test_set_size = len(X_test) / data_size

    # define pie chart 
    plt.figure(figsize=(7,7))
    pie_colors = ['seagreen', 'gold', 'tomato']
    plt.pie(
        [train_set_size, valid_set_size, test_set_size],
        labels=['Train Set', 'Validation Set', 'Test Set'],
        colors=pie_colors,
        autopct='%1.0f%%'
    )
    plt.title('Data Split Proportion')
    
    # save plot
    path_to_save = f'{path_to_output}/data_split_pie_chart.png'
    plt.savefig(path_to_save)
    
    print(f'Data split pie chart saved in {path_to_save}')


'''
function to save wine quality histogram of y_train
'''
def save_wine_quality_hist(y_train, path_to_output):
    # define histogram
    plt.figure(figsize=(7,7))
    y_train.value_counts().sort_index().plot(kind="bar")
    plt.title("Distribution of Wine Quality")
    plt.xlabel("Quality")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    
    # save plot
    path_to_save = f'{path_to_output}/wine_quality_histogram.png'
    plt.savefig(path_to_save)
    
    print(f'Wine quality histogram saved in {path_to_save}')


'''
function to save feature distributions of X_train
'''
def save_feature_dist(X_train, path_to_output):
    # define histograms
    X_train.hist(bins=20, figsize=(12,10))
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    
    # save plot
    path_to_save = f'{path_to_output}/feature_distributions.png'
    plt.savefig(path_to_save)
    
    print(f'Feature distributions saved in {path_to_save}')


'''
function to save correlation matrix 
'''
def corr_mat(X_train, y_train, path_to_output):
    # define corr mat
    corr = pd.concat([X_train, y_train], axis=1).corr()
    
    plt.figure(figsize=(7,7))
    plt.imshow(corr, cmap="coolwarm", interpolation="none")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    
    # save plot
    path_to_save = f'{path_to_output}/correlation_matrix.png'
    plt.savefig(path_to_save)
    
    print(f'Correlation Matrix saved in {path_to_save}')


if __name__ == "__main__":
    # define parser and add args
    parser = argparse.ArgumentParser(description='Save tables and figures exploratory data visualizations and tables')
    parser.add_argument('--path_to_processed_data', type=str, help='Path to train, validation, and test data')
    parser.add_argument('--path_to_output', type=str, help='Path to save plots and tables')

    # parse args
    args = parser.parse_args()
    
    # load X_train, X_valid, X_test, y_train
    X_train = pd.read_csv(f'{args.path_to_processed_data}/train_data/X_train.csv')
    y_train = pd.read_csv(f'{args.path_to_processed_data}/train_data/y_train.csv')
    
    X_valid = pd.read_csv(f'{args.path_to_processed_data}/valid_data/X_valid.csv')
    X_test = pd.read_csv(f'{args.path_to_processed_data}/test_data/X_test.csv')
    
    # save plots and tables
    save_feature_description(args.path_to_output)
    save_data_split_pie_chart(X_train, X_valid, X_test, args.path_to_output)
    save_wine_quality_hist(y_train, args.path_to_output)
    save_feature_dist(X_train, args.path_to_output)
    corr_mat(X_train, y_train, args.path_to_output)
    