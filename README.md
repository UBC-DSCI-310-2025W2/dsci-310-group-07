# White Wine Quality Prediction
## Authors
DSCI 310 GROUP 7:
- Justin Kim
- Aryan Arora
- Shuri Yamamoto
- Erhan Asad Javed

## Project Summary
This project focuses on multiclass classification of white wine quality based on wine characteristics such as acidity, density, and pH level, using a random forest classifier. This model could help wine producers predict white wine quality based on chemical characteristics, supporting quality control, production decisions, and pricing strategies. Our current model achieves a test accuracy of 64.8%, a precision of 34.5%, a recall of 49.5%, and an F1-score of 37.8%. Further details on the data analysis and prediction are available [here](reports/white_wine_quality_analysis.ipynb).

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

There are two ways to run the analysis:

### Docker Compose
1. Start the container with Docker Compose:
```bash
docker-compose up
```

2. Once the container has launched, open a web browser and go to:
```text
http://localhost:8888/lab
```

3. Open a terminal inside Jupyter Lab and run the full analysis pipeline:
```bash
conda activate project_env
make clean
make all
```

4. This will:
- download the raw dataset from the UCI Machine Learning Repository
- preprocess and split the data into training, validation, and test sets
- generate exploratory figures and tables
- train and evaluate the random forest classifier
- render the final report in HTML and PDF format

5. After the pipeline finishes, the main report outputs will be available at:
```text
reports/white_wine_quality_analysis.html
reports/white_wine_quality_analysis.pdf
```

6. After you finish working, stop the container by pressing **Ctrl + C** in the terminal. Then remove the container with:
```bash
docker-compose down
```

### Build Docker Image Locally
1. Build the Docker image:
```bash
docker build -t white-wine-analysis .
```

2. Start a container from the image:
```bash
docker run --rm -p 8888:8888 white-wine-analysis
```

3. Once the container has launched, open a web browser and go to:
```text
http://localhost:8888/lab
```

4. Open a terminal inside Jupyter Lab and run:
```bash
conda activate project_env
make clean
make all
```

5. This will generate the final analysis outputs, including:
```text
reports/white_wine_quality_analysis.html
reports/white_wine_quality_analysis.pdf
```

6. After you finish working, stop the container by pressing **Ctrl + C** in the terminal. Because the container was started with the `--rm` flag, it will be automatically removed when it stops.
