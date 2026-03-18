import argparse
import pandas as pd
import matplotlib.pyplot as plt

'''
function to save feature descripition table in the specified path
'''
def feature_description(path_to_save):
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
    df.to_csv(f'{path_to_save}/feature_description.csv')
    
    print(f'feature description table saved in {path_to_save}/feature_description.csv')
    
'''
function to save pie chart that presents data split ratio of train, validation, and test data
'''
def plot_data_split_pie_chart():
    pass

'''
function to save summary statistics of the train data
'''
def save_summary_stat():
    pass

'''
function to save wine quality histogram of y_train
'''
def save_wine_quality_hist():
    pass

'''
function to save feature distributions of X_train
'''
def save_feature_dist():
    pass

'''
function to save correlation matrix 
'''
def corr_mat():
    pass

if __name__ == "__main__":
    # define parser and add args
    parser = argparse.ArgumentParser(description='Save tables and figures exploratory data visualizations and tables')

    # TODO: figure out what args needed
    parser.add_argument()

    # parse args
    args = parser.parse_args()
    
    # TODO: call functions above after args and function definitions