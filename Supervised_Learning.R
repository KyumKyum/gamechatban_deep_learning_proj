# AI-X Final Project
# Supervised Learning after doing unsupervised learning.

# Dataset: A chatlog with toxic score assessed by unsupervised learning.
# Column: X(id), Message, Most common report reason, Toxic Score 

library(keras) # R Package: LSTM Model
library(caret) # R Package: 
library(tensorflow) # R Package: Tensorflow
library(reticulate) # R Package for interfacing with python
library(lightgbm) # R PAckage: LightGBM

setwd("~/Desktop/Dev/HYU/2023-02/AI-X/project/gamechatban") # Change this value to your working dir.

#use_virtualenv("/Users/kyumericano/.virtualenvs/r-reticulate/bin/python", required = TRUE)
reticulate::py_config() # Show python interpreter currently configured to use.

# Read a dataset for supervised learning.
processed_df <- read.csv('./offender_chatlog_with_toxic_score.csv')

# We are trying to achieve two objective by following supervised learning.
# 1.Build a model that can predict (regress) the toxic level of message.
# 2.Build a model that can classify the most possible common report reason of following chat.

# Objective 1. [Regression Task]: Build a model that can predict the toxic level of message.
# Strategy
# 1. Feature Engineering
  # Extract message and toxic score, which are the features required for learning.
  # Split data into training data and test data (75% : 25%)
# 2. Build a regression model using LSTM Model
  # Make regression and evaluate the model.
  # Plot the model result.

# 1-1 Feature Angineering
# Remove the missing values
processed_df <- na.omit(processed_df)
# Remove rows where toxic_score is equal to 0
processed_df <- processed_df[processed_df$toxic_score != 0, ] 

# Split the data
# Assuming your dataframe is named processed_df
# Make sure to replace 'toxic_score' with the actual name of your target variable
target_variable <- "toxic_score"

# Preprocess the text data for LSTM
tokenizer <- text_tokenizer()
fit_text_tokenizer(tokenizer, processed_df$message)
sequences <- texts_to_sequences(tokenizer, processed_df$message)
X <- pad_sequences(sequences, maxlen = 50L)  # The maximum length of message is 48.

# Split the data into training and testing sets
set.seed(sample(100:1000,1,replace=F)) # Random sampling
sample_index <- sample(1:nrow(processed_df), 0.8 * nrow(processed_df))
train_data <- X[sample_index, ]
test_data <- X[-sample_index, ]
train_labels <- processed_df$toxic_score[sample_index]
test_labels <- processed_df$toxic_score[-sample_index]

unique_words <- unique(unlist(strsplit(tolower(processed_df$message), " ")))
vocabulary_size <- length(unique_words)
embedding_dim <- round(sqrt(vocabulary_size))

# Build the LSTM model
model <- keras_model_sequential()
model %>%
  layer_embedding(input_dim = vocabulary_size, output_dim = embedding_dim, input_length = 50L) %>%
  layer_lstm(units = 100) %>%
  layer_dense(units = 1)

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'mean_squared_error',
  metrics = c('mean_absolute_error')
)

# Train the model
lstm_start_time <- Sys.time()


history <- model %>% fit(
  train_data, train_labels,
  epochs = 10, batch_size = 32,
  validation_split = 0.2
)

lstm_end_time <- Sys.time()

lstm_elapsed_time <- lstm_end_time - lstm_start_time
cat("Training Time (LSTM): ", lstm_elapsed_time, "\n")


# Evaluate the model
model %>% evaluate(test_data, test_labels)

# Visualization
plot(history$metrics$loss, type = "l", col = "blue", xlab = "Epoch", ylab = "Loss", main = "Training and Validation Loss")
lines(history$metrics$val_loss, col = "red")
legend("topright", legend = c("Training Loss", "Validation Loss"), col = c("blue", "red"), lty = 1:1)

# 1-3: Build a regression model using LSTM model. (Long Short Term Memory Model)


###### LightDBM (Gradient Boosting Model)

# LightGBM Model for Regression
lgb_data <- lgb.Dataset(train_data, label = train_labels)

# Set LightGBM parameters
lgb_params <- list(
  objective = "regression",  # Use "regression" for regression tasks
  metric = "rmse",  # Root Mean Squared Error as the evaluation metric
  num_iterations = 100
)

# Train the model
lgb_start_time <- Sys.time()

lgb_model <- lgb.train(params = lgb_params, data = lgb_data, verbose = 1)

lgb_end_time <- Sys.time()

lgb_elapsed_time <- lgb_end_time - lgb_start_time
cat("Training Time (LSTM): ", lgb_elapsed_time, "\n")

# Make predictions on the test set
predictions <- predict(lgb_model, test_data)

# Evaluate the model
mse <- mean((predictions - test_labels)^2)
rmse <- sqrt(mse)
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

## Visualization
# Calculate residuals
residuals <- predictions - test_labels
qqnorm(residuals)
qqline(residuals, col = "red")

