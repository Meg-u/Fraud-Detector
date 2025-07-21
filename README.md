Task 1: Data Analysis & Preprocessing
This task focuses on preparing e-commerce and bank transaction data for fraud detection modeling.

Objectives
Clean and preprocess the datasets
Handle missing values and duplicates
Engineer meaningful features
Merge geolocation data from IP address mapping
One-hot encode categorical variables
Handle class imbalance for fraud classification

Files Used
Fraud_Data.csv: E-commerce transaction data
IpAddress_to_Country.csv: Maps IP address ranges to countries
creditcard.csv: Bank card transaction data

Key Steps
Data Cleaning
Dropped duplicates
Filled missing values in sex, browser, and source with "Unknown"
Converted timestamps and IPs to usable formats
Feature Engineering
Extracted hour_of_day, day_of_week, time_since_signup
Added transaction_count per user
Mapped IP addresses to countries
Categorical Encoding
Applied one-hot encoding to categorical features
Class Imbalance Handling
Used SMOTE to oversample minority class in training data

Output
Cleaned and processed dataset saved to:
data/processed/fraud_data_processed.csv
