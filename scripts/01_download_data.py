import argparse

from dsci310_2025w2_winequalitypy.download_utils import download_data

if __name__ == "__main__":
    # define parser and add args
    parser = argparse.ArgumentParser(description='Download data from UCI repo and save it in a local folder')
    parser.add_argument('--uci_id', type=int, help='UCI repo dataset ID')
    parser.add_argument('--path_to_save', type=str, help='path to save downloaded data')
    
    # parse args
    args = parser.parse_args()
    
    # download data
    download_data(args.uci_id, args.path_to_save)