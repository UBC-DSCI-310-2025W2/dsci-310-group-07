import argparse
import pandas as pd
from ucimlrepo import fetch_ucirepo 

'''
function to donwload data from UCI repository with the given dataset ID and save it in the specified path
'''
def download_data(uci_id, path_to_save):
    # fetch dataset
    wine_quality = fetch_ucirepo(id=uci_id)

    # features and target
    X = wine_quality.data.features
    y = wine_quality.data.targets

    # concat features and target
    df = pd.concat([X, y], axis=1)
    
    # save df to the given path
    df.to_csv(path_to_save, index=False)
    
    print(f'File loaded from UCI repo with {uci_id} ID | File saved in {path_to_save}')
    
if __name__ == "__main__":
    # define parser and add args
    parser = argparse.ArgumentParser(description='Download data from UCI repo and save it in a local folder')
    parser.add_argument('--uci_id', type=int, help='UCI repo dataset ID')
    parser.add_argument('--path_to_save', type=str, help='path to save downloaded data')
    
    # parse args
    args = parser.parse_args()
    
    # download data
    download_data(args.uci_id, args.path_to_save)