# AI-X Final Project
# Supervised Learning after doing unsupervised learning.
# Part 2: Classifying Common Report Reason of Chatting

# Dataset: A chatlog with toxic score assessed by unsupervised learning.
# Column: X(id), Message, Most common report reason, Toxic Score 

# Load required libraries
library(randomForest) # Random Forest Model
library(caret) # Confusion Matrix
library(pROC) # pROC Curve


setwd("~/Desktop/Dev/HYU/2023-02/AI-X/project/gamechatban") # Change this value to your working dir.

# Read a dataset for supervised learning.
processed_df <- read.csv('./offender_chatlog_with_toxic_score.csv')
# Remove missing value
sampled_df <- processed_df[complete.cases(processed_df[,c("message", "most_common_report_reason")]),]

# Factorize the target variable 'most common report reason'.
sampled_df$most_common_report_reason = as.factor(sampled_df$most_common_report_reason)

# Split the data into training and testing sets
set.seed(sample(100:1000,1,replace=F)) # Random sampling
trainIndex <- createDataPartition(sampled_df$most_common_report_reason, p = 0.8, list = FALSE)
train_data <- sampled_df[trainIndex, ]
test_data <- sampled_df[-trainIndex, ]

# Make sure the columns match between train_data and test_data
train_features <- setdiff(names(train_data), "most_common_report_reason")
test_features <- setdiff(names(test_data), "most_common_report_reason")

# Model: Random Forest Classifier
rf_start_time <- Sys.time()

# Train a Random Forest model
rf_model <- randomForest(most_common_report_reason ~ ., data = train_data, ntree = 100)

# Check ending time
rf_end_time <- Sys.time()

# Calcualte Elapsed time;
rf_elapsed_time <- rf_end_time - rf_start_time
cat("Training Time (SVM): ", rf_elapsed_time, "\n")

# Make predictions on the test set
rf_predictions <- predict(rf_model, newdata = test_data)

# Evaluate the Random Forest model
confusion_matrix_rf <- table(Actual = test_data$most_common_report_reason, Predicted = rf_predictions)
accuracy_rf <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)

# Display confusion matrix and accuracy for Random Forest
print("Confusion Matrix (Random Forest):")
print(confusion_matrix_rf)
cat("Accuracy (Random Forest):", accuracy_rf, "\n")

# Visualization (Confusion Matrix Plot for Random Forest)
conf_mat_rf <- confusionMatrix(rf_predictions, test_data$most_common_report_reason)
# Extract confusion matrix values
conf_matrix_values <- conf_mat_rf$table

# Plot the confusion matrix
heatmap(conf_matrix_values, 
        col = c("white", "lightblue", "blue"), 
        main = "Confusion Matrix (Random Forest)",
        xlab = "Predicted",
        ylab = "Actual")

# ROC Curve
#The Receiver Operating Characteristic (ROC) curve is a graphical representation that illustrates the performance of a binary classification model across different discrimination thresholds. 
# It plots the True Positive Rate (Sensitivity) against the False Positive Rate (1 - Specificity) for various threshold values.
# Make predictions on the test set
rf_probs <- as.numeric(predict(rf_model, newdata = test_data, type = "response"))

# Create binary labels for ROC curve
actual_labels_binary <- as.numeric(test_data$most_common_report_reason) - 1  # Assuming binary classification

# Create ROC curve
roc_curve <- roc(actual_labels_binary, rf_probs)

# Plot the ROC curve
plot(roc_curve, col = "blue", main = "ROC Curve for Random Forest",
     col.main = "darkblue", col.lab = "black", lwd = 2)


