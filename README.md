# White Wine Quality Prediction
## Authors
DSCI 310 GROUP 7:
- Justin Kim
- Aryan Arora
- Shuri Yamamoto
- Erhan Asad Javed

## Project Summary
This project focuses on multiclass classification of white wine quality based on wine characteristics such as acidity, density, and pH level, using a random forest classifier. This model could help wine producers predict white wine quality based on chemical characteristics, supporting quality control, production decisions, and pricing strategies. Our current model achieves a test accuracy of 64.8%, a precision of 34.5%, a recall of 49.5%, and an F1-score of 37.8%. Further details on the data analysis and prediction are available in the [Jupyter notebook](reports/white_wine_quality_analysis.ipynb) or the [Quarto PDF report](reports/white_wine_quality_analysis.pdf). You can also preview the Quarto document to view the results by following the [instructions](#instructions-on-running-the-data-analysis) provided in the later section.

## Dataset
We use `winequality-white.csv` dataset from the [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/dataset/186/wine+quality) for white wine data analysis and quality prediction. The dataset contains 4898 samples of different white variants of the Portuguese Vinho Verde wine. 

## Dependencies
We use Python version 3.14.3 as the primary programming language for data analysis and prediction. Moreover, the list of Jupyter and Python packages, along with their versions, is provided in [environment.yml](environment.yml).

## Instructions on Running the Data Analysis
We configure the computational environment used in this project to be reproducible through containerization with Docker. To run the data analysis, please ensure that Docker is installed on your system. If you do not already have Docker installed, you can download it from [here](https://www.docker.com/).

Once Docker is installed and running, follow the steps below.

1. Clone the repository:
```bash
git clone https://github.com/UBC-DSCI-310-2025W2/dsci-310-group-07.git
```

2. Navigate to the root directory of the repository:
```bash
cd dsci-310-group-07/
```

3. Start the container with Docker Compose:
```bash
docker compose run --rm -p 8888:8888 project
```
- If you are using Git Bash on Windows, use the following command instead:
```bash
winpty docker compose run --rm -p 8888:8888 project
```

4. Once the container has launched, you will be placed inside an interactive Bash shell within the container. Please follow the instructions below to run Make, Quarto, and Jupyter.

<br>

### Using Make
This project uses a [Makefile](Makefile) to automate the full data analysis and reporting pipeline.

#### Running the Full Pipeline
To run the full pipeline for data analysis, model training, prediction, and report generation, use the following command:

```bash
make all
```

This will:
1. Download the raw dataset from the UCI Machine Learning Repository
2. Preprocess and split the data into training, validation, and test sets
3. Perform exploratory data analysis (EDA) and generate figures and tables
4. Train and evaluate the Random Forest classifier
5. Generate the final Quarto report in both HTML and PDF formats

#### Resetting the Pipeline
To reset the project to a clean state, use the following command:

```bash
make clean
```
This removes all generated data, EDA and prediction outputs, and Quarto reports. Note that you must run `make all` again before previewing the Quarto documents as described below.

<br>

### Quarto 
#### PDF & HTML Reports
After running `make all`, the Quarto reports (PDF and HTML) can be found in the `reports/` folder.

#### Quarto Preview
To interactively view the Quarto document:

1. Run the following preview command from the project root:
```bash
quarto preview reports/white_wine_quality_analysis.qmd --host 0.0.0.0 --port 8888
```

2. Click the URL displayed in the terminal, or open a web browser and navigate to the following URL:
```
http://localhost:8888
```

3. To stop previewing, press Ctrl + C in the terminal.

<br>

### Running the Jupyter Notebook 
1. Run the following command to start JupyterLab:
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --IdentityProvider.token=''
```
2. Click the URL displayed in the terminal (e.g., http://127.0.0.1:8888/lab) or pasting it into a web browser.
3. Open `reports/white_wine_quality_analysis.ipynb`, then click Kernel > Restart Kernel and Run All Cells to execute the analysis.
4. To stop JupyterLab, press Ctrl + C in the terminal

<br>

### After Finishing Work
1. Exit the container by running the following command in the Bash shell inside the container:
```bash
exit
```
2. Then, run the following command:
```
docker compose down
```

## Licenses
The project report is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/). The software and code in the project are licensed under the [MIT License](https://opensource.org/licenses/MIT). Further details regarding the licenses are provided in [LICENSE.md](LICENSE.md).
