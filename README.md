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

## Instructions on running data analysis
We configure the computational environment used in this project to be reproducible through containerization with Docker. To run the data analysis, please ensure that Docker is installed on your system. If you don't have Docker installed, you can download it from [here](https://www.docker.com/). 

Once Docker is installed, start Docker and follow the instructions below to run the data analysis:

1. Clone the repository using the following command:
```
git clone https://github.com/UBC-DSCI-310-2025W2/dsci-310-group-07.git
```

2. Navigate to the root directory of the repository:
```
cd dsci-310-group-07/
```
<br>
Then, there are two ways to run the data analysis:

### Docker Compose
1. Run the following command:
```
docker-compose up
```

2. Once the container has launched, open a web browser and navigate to the following URL to open Jupyter Lab:
```
http://localhost:8888/lab
```

3. Open `reports/white_wine_quality_analysis.ipynb`, then click **Kernel > Restart Kernel and Run All Cells** to execute the analysis.

4. After you finish working, stop the container by pressing **Ctrl + C** in the terminal. Then, run the following command to remove the container:
```
docker-compose down
```

### Build Docker Image Locally
1. Build the Docker image:
```
docker build -t <environment-name> .
```

2. After building the image, start a container:
```
docker run --rm -p 8888:8888 <environment-name>
```

3. Once the container has launched, open a web browser and navigate to the following URL to open Jupyter Lab:
```
http://localhost:8888/lab?
```

4. Open `reports/white_wine_quality_analysis.ipynb`, then click **Kernel > Restart Kernel and Run All Cells** to execute the analysis.

5. After you finish working, stop the container by pressing **Ctrl + C** in the terminal. Because the container was started with the `--rm` flag, it will be automatically removed when it stops.

## Licenses
The project report is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/). The software and code in the project are licensed under the [MIT License](https://opensource.org/licenses/MIT). Further details regarding the licenses are provided in [LICENSE.md](LICENSE.md).


