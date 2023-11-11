# AI-X Final Project
# Unsupervised Learning based on TF-IDF

# Dataset: League of Legends Tribunal Chatlogs (Kaggle)
# https://www.kaggle.com/datasets/simshengxue/league-of-legends-tribunal-chatlogs

library(dplyr) # R Package: dplyr - advanced filtering and selection
library(tm) # R Package: tm - Text Mining/Merging for preprocess of TF-IDF, and TF-IDF itself

setwd("~/Desktop/Dev/HYU/2023-02/AI-X/project/gamechatban") # Change this value to your working dir.

chatlogs <- read.csv("./chatlogs.csv")

# Pre-Processing Steps; Feature Engineering Pipeline for the chatlogs. 
# 1. Grammatical & Game-specific Expression Removal:
  # Remove common grammatical expressions like "is" and "are" to enhance the validity of the analysis.
# 2. Feature Engineering - Severity:
  # Introduce a new feature called 'severity' based on the total number of case reports.
  # Total case report <= 3: Severe
  # Total Case Report >= 4 && <= 6: Normal
  # Total Case Report >= 7: Low
# 3. Concatenation of Chatlogs:
  # Group chatlogs based on the common reported reason.
  # Concatenate chatlogs within each group into a single text.
  # Chatlogs will be merged into a single column for each most common reported reason, considering the newly defined 'severity' feature.

# Step 1: Gramatical Game-specific Expression Removal: Used gsub and REGEX to do such task.
# Read champion names
champion_names <- read.csv("./champion_names.csv")
# Create a regex pattern for both grammatical expressions and champion names/abbreviations
pattern <- paste0("\\b(?:is|are|&gt|&lt|was|were|", paste(unique(c(champion_names$Champion, champion_names$Abbreviation)), collapse = "|"), ")\\b")

# Remove both grammatical expressions and champion names/abbreviations from chatlogs$message
chatlogs$message <- gsub(pattern, "", chatlogs$message, ignore.case = TRUE)

# Export into csv for later use. (Pre-processed.csv)
write.csv(chatlogs, "processed.csv")

# Step 2:  Feature Engineering - Severity
chatlogs$severity <- cut( # Categorize numbers into factors.
  chatlogs$case_total_reports,
  breaks = c(-Inf, 3, 6, Inf),
  labels = c("Severe", "Normal", "Low"),
  include.lowest = TRUE
)

# Step 3: Concatenation of Chatlogs
concatenated <- chatlogs %>%
  group_by(most_common_report_reason, severity) %>% #Group by following two category
  summarise(concatenated_text = paste(message, collapse = " ")) %>%
  ungroup()

# TF-IDF (Term-Frequency Inverse Document Frequency) Matrix Anaylsis; Process TF-IDF for each concatenated text to get 'toxiticy level of each words'.
# 1. Create a corpus for TF-IDF, pre-process it.
# 2. Create DTM for TF-IDF, and generate TF-IDF matrix
# 3. Transpose it and apply new column name to analyse the reported reason and severity.
# 4. Scale Up and round the value to get toxic level.
# 5. Export into csv. 'toxicity_lev.csv'

# Create a Corpus from the column concatenated_text.
corpus <- Corpus(VectorSource(concatenated$concatenated_text))

# Additional pre-process of the text in the corpus. (e.g. removing punctuation, stripping whitespaces, etc.)
corpus <- tm_map(corpus, content_transformer(tolower)) # Convert each contents into lower case
corpus <- tm_map(corpus, removePunctuation) # Remove Punctuations
corpus <- tm_map(corpus, removeWords, stopwords("english")) # Remove Additional English stopwords (a, the, etc) that hadn't been filtered.
corpus <- tm_map(corpus, stripWhitespace) # Strip Whitespace

# Create DTM (Document-Term Matrix) based on the corpus, which is used for TF-IDF
dtm <- DocumentTermMatrix(corpus)

# Create TF-IDF Matrix based on the DTM.
tf_idf <- weightTfIdf(dtm)
tf_idf <- t(as.matrix(tf_idf)) # Transpose

# Generate Column name
tf_idf_col_name <- paste(concatenated$most_common_report_reason, concatenated$severity, sep = "_")

# Set the column name of the transposed TF_IDF
colnames(tf_idf) <- tf_idf_col_name

# Scale up and Round the values
tf_idf <- round((tf_idf * 10000), 2)

# Convert TF-IDF matrix into a new data frame for further analysis.
tf_idf_df <- as.data.frame(tf_idf)

# Make it into csv for further analysis & supervised learning.
write.csv(tf_idf_df, "toxicity_lev.csv")
