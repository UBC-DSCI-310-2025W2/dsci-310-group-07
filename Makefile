all: reports/white_wine_quality_analysis.html reports/white_wine_quality_analysis.pdf

# create folders for the analysis pipeline
data:
	mkdir -p data

data/processed_data:
	mkdir -p data/processed_data/train_data
	mkdir -p data/processed_data/valid_data
	mkdir -p data/processed_data/test_data

results:
	mkdir -p results

# 1: download raw data from UCI repository
data/wine_quality_raw.csv: scripts/download_data.py | data
	python scripts/download_data.py --uci_id=186 --path_to_save=data/wine_quality_raw.csv

# 2: preprocess and split data into train/validation/test sets
data/processed_data/train_data/X_train.csv \
data/processed_data/train_data/y_train.csv \
data/processed_data/valid_data/X_valid.csv \
data/processed_data/valid_data/y_valid.csv \
data/processed_data/test_data/X_test.csv \
data/processed_data/test_data/y_test.csv: scripts/preprocess_data.py data/wine_quality_raw.csv | data/processed_data
	python scripts/preprocess_data.py --path_to_raw_data=data/wine_quality_raw.csv --path_to_processed_data=data/processed_data

# 3: EDA tables and figures
results/feature_description.csv \
results/data_split_pie_chart.png \
results/wine_quality_histogram.png \
results/feature_distributions.png \
results/correlation_matrix.png: scripts/eda.py \
data/processed_data/train_data/X_train.csv \
data/processed_data/train_data/y_train.csv \
data/processed_data/valid_data/X_valid.csv \
data/processed_data/test_data/X_test.csv | results
	python scripts/eda.py \
		--path_to_processed_data=data/processed_data \
		--path_to_feature_description=results/feature_description.csv \
		--path_to_pie_chart=results/data_split_pie_chart.png \
		--path_to_wine_quality_hist=results/wine_quality_histogram.png \
		--path_to_feature_dist=results/feature_distributions.png \
		--path_to_corr_mat=results/correlation_matrix.png

# 4: train model and save outputs
results/best_model_parameters.csv \
results/test_prediction.csv \
results/confusion_matrix.png: scripts/train_and_evaluate.py \
data/processed_data/train_data/X_train.csv \
data/processed_data/train_data/y_train.csv \
data/processed_data/valid_data/X_valid.csv \
data/processed_data/valid_data/y_valid.csv \
data/processed_data/test_data/X_test.csv \
data/processed_data/test_data/y_test.csv | results
	python scripts/train_and_evaluate.py \
		--path_to_processed_data=data/processed_data \
		--path_to_prediction_data=results/test_prediction.csv \
		--path_to_model_params=results/best_model_parameters.csv \
		--path_to_conf_mat=results/confusion_matrix.png

# 5: render quarto report in html
reports/white_wine_quality_analysis.html: reports/white_wine_quality_analysis.qmd \
results/feature_description.csv \
results/data_split_pie_chart.png \
results/wine_quality_histogram.png \
results/feature_distributions.png \
results/correlation_matrix.png \
results/best_model_parameters.csv \
results/test_prediction.csv \
results/confusion_matrix.png
	quarto render reports/white_wine_quality_analysis.qmd --to html

# 6: render the quarto report in pdf (do we need this?)
reports/white_wine_quality_analysis.pdf: reports/white_wine_quality_analysis.qmd \
results/feature_description.csv \
results/data_split_pie_chart.png \
results/wine_quality_histogram.png \
results/feature_distributions.png \
results/correlation_matrix.png \
results/best_model_parameters.csv \
results/test_prediction.csv \
results/confusion_matrix.png
	quarto render reports/white_wine_quality_analysis.qmd --to pdf

# remove all the files
clean:
	rm -rf data
	rm -rf results
	rm -f reports/white_wine_quality_analysis.html
	rm -f reports/white_wine_quality_analysis.pdf
	rm -rf reports/white_wine_quality_analysis_files

.PHONY: all clean