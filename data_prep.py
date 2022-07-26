import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# read the dataset
dataset = pd.read_csv('dataset/adult.csv', na_values=['?'])

# Data Cleaning below

# Bucket low-freq categories as "Other"
dataset['native-country'] = ['United-States ' if x == 'United-States' else 'Other' for x in dataset['native-country']]

# categorical data encoding. Our target: income--> binary
ordinal_encoder = OrdinalEncoder()

income = dataset[['income']]
income_encoded = ordinal_encoder.fit_transform(income)
dataset['income'] = income_encoded

workclass = dataset[['workclass']]
workclass_encoded = ordinal_encoder.fit_transform(workclass)
dataset['workclass'] = workclass_encoded

education = dataset[['education']]
education_encoded = ordinal_encoder.fit_transform(education)
dataset['education'] = education_encoded

marital = dataset[['marital-status']]
marital_encoded = ordinal_encoder.fit_transform(marital)
dataset['marital-status'] = marital_encoded

occupation = dataset[['occupation']]
occupation_encoded = ordinal_encoder.fit_transform(occupation)
dataset['occupation'] = occupation_encoded

relationship = dataset[['relationship']]
relationship_encoded = ordinal_encoder.fit_transform(relationship)
dataset['relationship'] = relationship_encoded

race = dataset[['race']]
race_encoded = ordinal_encoder.fit_transform(race)
dataset['race'] = race_encoded

gender = dataset[['gender']]
gender_encoded = ordinal_encoder.fit_transform(gender)
dataset['gender'] = gender_encoded

country = dataset[['native-country']]
country_encoded = ordinal_encoder.fit_transform(country)
dataset['native-country'] = country_encoded


# Handling missing values, specifically "NaN" by impute missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer = imputer.fit(dataset)
dataset.iloc[:,:] = imputer.transform(dataset)

# Assign X as a DataFrame of features and y as a Series of the targets
X = dataset.drop('income', 1)
y = dataset.income

# Split training/test set after the data is completely cleaned up
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)