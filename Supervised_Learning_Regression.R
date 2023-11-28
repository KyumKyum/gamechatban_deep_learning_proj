# AI-X Final Project
# Supervised Learning after doing unsupervised learning.
# Part 1: Regressing Toxic Score of Chatting

# Dataset: A chatlog with toxic score assessed by unsupervised learning.
# Column: X(id), Message, Most common report reason, Toxic Score 

library(keras) # R Package: LSTM Model
library(caret) # R Package: 
library(tensorflow) # R Package: Tensorflow
library(reticulate) # R Package for interfacing with python
library(lightgbm) # R PAckage: LightGBM

setwd("~/Desktop/Dev/HYU/2023-02/AI-X/project/gamechatban") # Change this value to your working dir.

reticulate::py_config() # Show python interpreter currently configured to use.

# Read a dataset for supervised learning.
processed_df <- read.csv('./offender_chatlog_with_toxic_score.csv')

# Objective: [Regression Task]: Build a model that can predict the toxic level of message.
# Strategy
# 1. Feature Engineering
  # Extract message and toxic score, which are the features required for learning.
  # Split data into training data and test data (75% : 25%)
# 2. Build a regression model using LSTM Model
  # Make regression and evaluate the model.
  # Plot the model result.
# 3. Build a regression model using LightGBM.
  # Make regression and evaluate the model.
  # Plot the model result.
# 4. Analyze & Compare the result of two models,

# 1-1 Feature Engineering
# Remove the missing values
processed_df <- na.omit(processed_df)
# Remove rows where toxic_score is equal to 0
processed_df <- processed_df[processed_df$toxic_score != 0, ] 

# Preprocess the text data for LSTM
# Create a text tokenizer; convert text data into a format that can be fed into a neural network.
tokenizer <- text_tokenizer()

# Fit the text tokenizer on the messages in the 'processed_df' dataframe
# Learns the vocabulary of the corpus and assigns a unique integer index to each word.
fit_text_tokenizer(tokenizer, processed_df$message)

# Convert the text messages to sequences of integers using the fitted tokenizer
# Each word in the messages is replaced with its corresponding integer index.
sequences <- texts_to_sequences(tokenizer, processed_df$message)

# Pad the sequences to ensure uniform length (Maximum length of 50 tokens)
X <- pad_sequences(sequences, maxlen = 50L)  # The maximum length of message is 48.

# Split the data into training and testing sets
set.seed(sample(100:1000,1,replace=F)) # Random sampling
sample_index <- sample(1:nrow(processed_df), 0.8 * nrow(processed_df))
train_data <- X[sample_index, ]
test_data <- X[-sample_index, ]
train_labels <- processed_df$toxic_score[sample_index]
test_labels <- processed_df$toxic_score[-sample_index]

# Build the LSTM model
# Create a sequential model
model <- keras_model_sequential()

# Add an embedding layer to convert integer sequences to dense vectors
  # 'input_dim' is the size of the vocabulary (output of the tokenizer)
  # 'output_dim' is the dimension of the dense embedding
  # 'input_length' is the length of the input sequences (padded to 50 tokens)

unique_words <- unique(unlist(strsplit(tolower(processed_df$message), " "))) # Unique words of each message.

vocabulary_size <- length(unique_words) # Size of vocab. Will be a value of input_dim.
embedding_dim <- round(sqrt(vocabulary_size)) # Size of output, Will be a root(sqrt) of vocab size,

# Adding embedding layer based on the calcualted vocab size and embedding size.
model %>%
  layer_embedding(input_dim = vocabulary_size, output_dim = embedding_dim, input_length = 50L) %>%
  layer_lstm(units = 100) %>%
  layer_dense(units = 1)

# Compile the model
model %>% compile(
  optimizer = 'adam', # Adam Optimizer
  loss = 'mean_squared_error',  # Mean Squared Error loss for regression
  metrics = c('mean_absolute_error') # Mean Absolute Error as an additional metric
)

# Train the model 
# Check starting time
lstm_start_time <- Sys.time()

# Start Training
history <- model %>% fit(
  train_data, train_labels,
  epochs = 10, batch_size = 32,
  validation_split = 0.2
)

# Check ending time
lstm_end_time <- Sys.time()

# Calcualte Elapsed time;
lstm_elapsed_time <- lstm_end_time - lstm_start_time
cat("Training Time (LSTM): ", lstm_elapsed_time, "\n")

# Evaluate the model
model %>% evaluate(test_data, test_labels)

# Visualization: Line Plot of Training and Validation Loss
plot(history$metrics$loss, type = "l", col = "blue", xlab = "Epoch", ylab = "Loss", main = "Training and Validation Loss")
lines(history$metrics$val_loss, col = "red")
legend("topright", legend = c("Training Loss", "Validation Loss"), col = c("blue", "red"), lty = 1:1)

# 1-3: Build a regression model using LightGBM (Gradient Boosting Model)

# LightGBM Model for Regression
lgb_data <- lgb.Dataset(train_data, label = train_labels)

# Set LightGBM parameters
lgb_params <- list(
  objective = "regression",  # Use "regression" for regression tasks
  metric = "rmse",  # Root Mean Squared Error as the evaluation metric
  num_iterations = 100 # Number of iteration are going to apply.
)

# Train the model 
# Check starting time
lgb_start_time <- Sys.time()

# Start Training
lgb_model <- lgb.train(params = lgb_params, data = lgb_data, verbose = 1)

# Check ending time
lgb_end_time <- Sys.time()

# Calcualte Elapsed time;
lgb_elapsed_time <- lgb_end_time - lgb_start_time
cat("Training Time (LSTM): ", lgb_elapsed_time, "\n")

# Make predictions on the test set
predictions <- predict(lgb_model, test_data)

# Evaluate the model (RMSE)
mse <- mean((predictions - test_labels)^2)
rmse <- sqrt(mse)
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

## Visualization
# Calculate residuals
residuals <- predictions - test_labels

# Q-Q (Quantile-Quantile) Plot of Residuals
# Q-Q plots are used to assess whether a set of data follows a particular theoretical distribution, such as a normal distribution.
# In this context, the plot compares the quantiles of the residuals against the quantiles of a standard normal distribution. (norm)
# If the points on the plot closely follow the reference line, it suggests that the residuals are approximately normally distributed.
# Deviations from the line may indicate non-normality.(Potential issues exists with the assumptions of the regression model)
qqnorm(residuals)
#Add a straight line, which passes first and third quatiles of the data, for a reference line to the Q-Q plot.
qqline(residuals, col = "red")

