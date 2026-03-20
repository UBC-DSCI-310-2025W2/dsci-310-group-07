all: reports/white_wine_quality_analysis.html reports/white_wine_quality_analysis.pdf

# create folders for the analysis pipeline
data:
	mkdir -p data

data/raw:
	mkdir -p data/raw

data/processed:
	mkdir -p data/processed/train_data
	mkdir -p data/processed/valid_data
	mkdir -p data/processed/test_data
	mkdir -p data/processed/prediction

output:
	mkdir -p output

output/eda:
	mkdir -p output/eda

output/prediction:
	mkdir -p output/prediction

# 1: download raw data from UCI repository
data/raw/winequality-white.csv: scripts/download_data.py | data/raw
	python scripts/download_data.py --uci_id=186 --path_to_save=data/raw/winequality-white.csv
 
# 2: preprocess and split data into train/validation/test sets
data/processed/train_data/X_train.csv \
data/processed/train_data/y_train.csv \
data/processed/valid_data/X_valid.csv \
data/processed/valid_data/y_valid.csv \
data/processed/test_data/X_test.csv \
data/processed/test_data/y_test.csv: scripts/preprocess_data.py data/raw/winequality-white.csv | data/processed
	python scripts/preprocess_data.py --path_to_raw_data=data/raw/winequality-white.csv --path_to_processed=data/processed

# 3: EDA tables and figures
output/eda/feature_description.csv \
output/eda/data_split_pie_chart.png \
output/eda/wine_quality_histogram.png \
output/eda/feature_distributions.png \
output/eda/correlation_matrix.png: scripts/eda.py \
data/processed/train_data/X_train.csv \
data/processed/train_data/y_train.csv \
data/processed/valid_data/X_valid.csv \
data/processed/test_data/X_test.csv | output/eda
	python scripts/eda.py \
		--path_to_processed=data/processed \
		--path_to_feature_description=output/eda/feature_description.csv \
		--path_to_pie_chart=output/eda/data_split_pie_chart.png \
		--path_to_wine_quality_hist=output/eda/wine_quality_histogram.png \
		--path_to_feature_dist=output/eda/feature_distributions.png \
		--path_to_corr_mat=output/eda/correlation_matrix.png

# 4: train model and save outputs
output/prediction/best_model_parameters.csv \
data/processed/prediction/test_prediction.csv: scripts/train_and_evaluate.py \
data/processed/train_data/X_train.csv \
data/processed/train_data/y_train.csv \
data/processed/valid_data/X_valid.csv \
data/processed/valid_data/y_valid.csv \
data/processed/test_data/X_test.csv \
data/processed/test_data/y_test.csv | output/prediction
	python scripts/train_and_evaluate.py \
		--path_to_processed=data/processed \
		--path_to_prediction_data=data/processed/prediction/test_prediction.csv \
		--path_to_model_params=output/prediction/best_model_parameters.csv \
		--path_to_conf_mat=output/prediction/confusion_matrix.png

# 5: render quarto report in html
reports/white_wine_quality_analysis.html: reports/white_wine_quality_analysis.qmd \
output/eda/feature_description.csv \
output/eda/data_split_pie_chart.png \
output/eda/wine_quality_histogram.png \
output/eda/feature_distributions.png \
output/eda/correlation_matrix.png \
output/prediction/best_model_parameters.csv \
data/processed/prediction/test_prediction.csv \
output/prediction/confusion_matrix.png
	quarto render reports/white_wine_quality_analysis.qmd --to html

# 6: render the quarto report in pdf (do we need this?)
reports/white_wine_quality_analysis.pdf: reports/white_wine_quality_analysis.qmd \
output/eda/feature_description.csv \
output/eda/data_split_pie_chart.png \
output/eda/wine_quality_histogram.png \
output/eda/feature_distributions.png \
output/eda/correlation_matrix.png \
output/prediction/best_model_parameters.csv \
data/processed/prediction/test_prediction.csv \
output/prediction/confusion_matrix.png
	quarto render reports/white_wine_quality_analysis.qmd --to pdf

# remove all the files
clean:
	rm -rf data
	rm -rf output
	rm -f reports/white_wine_quality_analysis.html
	rm -f reports/white_wine_quality_analysis.pdf
	rm -rf reports/white_wine_quality_analysis_files

.PHONY: all clean