---
title: "R Notebook"
output: html_notebook
---


### Question 1 —
```{r}
#1.

library(readr)

# Load the dataset
data <- read.csv("/Users/nishabled/Desktop/Ms NEU/DA5020/diabetes.csv")
head(data)

# Summary statistics
summary(data)

```

### Question 2 —
```{r}
#2.

library(tidyverse)

# Normalize the explanatory variables using min-max normalization
normalized_diabetes <- data %>%
  select(-Outcome) %>%
  mutate(across(everything(), ~ (.-min(.)) / (max(.) - min(.))))

# Combine normalized explanatory variables with the response variable (Outcome)
normalized_diabetes <- cbind(normalized_diabetes, diabetes$Outcome)

# Inspect the first few rows of the normalized dataset
head(normalized_diabetes)

# Summary statistics of the normalized dataset
summary(normalized_diabetes)

```
```{r}
#3.

# Set seed for reproducibility
set.seed(123)

# Sample 80% of the rows for training set
train_index <- sample(1:nrow(normalized_data), 0.8 * nrow(normalized_data))

# Create training set
train_data <- normalized_data[train_index, ]

# Create test set
test_data <- normalized_data[-train_index, ]

# Check the dimensions of training and test sets
dim(train_data)
dim(test_data)


```


```{r}
#4.
# Define the distance function (Euclidean distance)
euclidean_distance <- function(x1, x2) {
  return(sqrt(sum((x1 - x2)^2)))
}

# Define the knn_predict function
knn_predict <- function(train.data, test.data, k) {
  # Initialize vector to store predictions
  predictions <- c()
  
  # Loop through each test observation
  for (i in 1:nrow(test.data)) {
    # Calculate distances between the test observation and all training observations
    distances <- apply(train.data[, -ncol(train.data)], 1, function(x) euclidean_distance(x, test.data[i, -ncol(test.data)]))
    
    # Combine distances with corresponding class labels
    neighbors <- cbind(distances, train.data[, ncol(train.data)])
    
    # Sort neighbors by distance
    neighbors <- neighbors[order(neighbors[, 1]), ]
    
    # Select the k nearest neighbors
    k_nearest_neighbors <- neighbors[1:k, 2]
    
    # Determine the majority class among the k nearest neighbors
    predicted_class <- ifelse(sum(k_nearest_neighbors) >= k/2, 1, 0)
    
    # Append the predicted class to the predictions vector
    predictions <- c(predictions, predicted_class)
  }
  
  # Return the vector of predictions
  return(predictions)
}

```


```{r}
#5.

# Example usage of knn_predict function with k = 6
knn_predict <- knn_predict(train.data = train_data, test.data = test_data, k = 6)
cat("knn_predict","\n",knn_predict, "\n")

# Analyze the results using a confusion matrix
conf_matrix <- table(Actual = test_data[, ncol(test_data)], Predicted = predicted_classes)
cat("conf_matrix k=6:", conf_matrix,"\n")

# Calculate accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("acuracy k=6:",accuracy,"\n")

```

```{r}
#6.

library(ggplot2)

# K values
k_values <- seq(1, 40, by = 2)

# Vector to store MSE for all k
mse_values <- numeric(length(k_values))

# Loop for k
for (i in seq_along(k_values)) {
  k <- k_values[i]
  mse_values[i] <- knn.predict(data_train = train_data, data_test = test_data, k = k)
}

# k values and their MSEs
data_frame_k_mse <- data.frame(k = k_values, MSE = mse_values)
print(data_frame_k_mse)


```

