# AI-X Final Project
# Unsupervised Learning based on TF-IDF

# Dataset: League of Legends Tribunal Chatlogs (Kaggle)
# https://www.kaggle.com/datasets/simshengxue/league-of-legends-tribunal-chatlogs

library(dplyr) # R Package: dplyr - advanced filtering and selection
library(tm) # R Package: tm - Text Merging for preprocess of TF-IDF

setwd("~/Desktop/Dev/HYU/2023-02/AI-X/project/gamechatban") # Change this value to your working dir.

chatlogs <- read.csv("./chatlogs.csv")

# Pre-Processing Steps; Feature Engineering Pipeline for the chatlogs. 
# 1. Grammatical Expression Removal:
  # Remove common grammatical expressions like "is" and "are" to enhance the validity of the analysis.
# 2. Feature Engineering - Severity:
  # Introduce a new feature called 'severity' based on the total number of case reports.
  # Total case report <= 2: Severe
  # Total Case Report >= 3 && <= 6: Normal
  # Total Case Report >= 7: Low
# 3. Concatenation of Chatlogs:
  # Group chatlogs based on the common reported reason.
  # Concatenate chatlogs within each group into a single text.
# 4. Merge into Single Column:
  # Merge the concatenated chatlogs into a single column for each common reported reason, considering the newly defined 'severity' feature.

# Step 1: Gramatical Expression Removal: Used gsub and REGEX to do such task.
chatlogs$message <- gsub("\\b(?:is|are)\\b", "", chatlogs$message, ignore.case = TRUE)

# Step 2:  Feature Engineering - Severity
chatlogs$severity <- cut( # Categorize numbers into factors.
  chatlogs$case_total_reports,
  breaks = c(-Inf, 2, 6, Inf),
  labels = c("Severe", "Normal", "Low"),
  include.lowest = TRUE
)

# Step 3: Concatenation of Chatlogs
concatenated <- chatlogs %>%
  group_by(most_common_report_reason, severity) %>% #Group by following two category
  summarise(concatenated_text = paste(message, collapse = " ")) %>%
  ungroup()

# Step 4: Merge into Single Column
merged_data <- concatenated %>%
  group_by(most_common_report_reason) %>%
  summarize(merged_text = paste(concatenated_text, collapse = " ")) %>%
  ungroup()
