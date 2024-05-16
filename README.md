# Diabetes Prediction Project

This project involves predicting the onset of diabetes in patients using the k-nearest neighbors (KNN) algorithm. The dataset used contains various health metrics and an outcome variable indicating whether a patient developed diabetes.

## Overview

The project aims to:

- Predict the onset of diabetes based on health metrics such as glucose level, blood pressure, BMI, etc.
- Implement the KNN algorithm for classification.
- Evaluate the model's performance using a confusion matrix and accuracy metric.
- Determine the optimal value of k for KNN through cross-validation.

## Data

The dataset used in this project, "diabetes.csv," contains the following columns:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (1: diabetes, 0: no diabetes)

## Analysis Steps

1. Load the dataset and examine the first few rows and summary statistics.
2. Normalize the explanatory variables using min-max normalization.
3. Split the dataset into training and test sets (80-20 split).
4. Implement the KNN algorithm to predict diabetes onset with various values of k.
5. Evaluate the model's performance using a confusion matrix and accuracy metric.
6. Determine the optimal value of k based on the mean squared error (MSE).

## Dependencies

- R programming language
- Required R packages: `readr`, `tidyverse`, `ggplot2`

## How to Use

To replicate the analysis:

1. Load the dataset "diabetes.csv" using the `read.csv` function.
2. Execute the provided R code chunks step by step.
3. Ensure that the required R packages are installed and loaded.
4. Customize the analysis as needed, such as modifying the value of k or adding additional preprocessing steps.
5. Analyze the results, including the confusion matrix and accuracy metric, to assess the model's performance.

## Results

The project generate predictions for diabetes onset based on the input health metrics. The accuracy of the model and the optimal value of k is determined, providing insights into the effectiveness of the KNN algorithm for diabetes prediction.

## Credits

- The dataset used in this project is sourced from a publicly available diabetes dataset.
- R Core Team and contributors for developing the R programming language and associated packages.
