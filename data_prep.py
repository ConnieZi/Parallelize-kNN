import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# Split train/test set
dataset = pd.read_csv('dataset/adult.csv', na_values=['#NAME?'])
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
# len(train_set) == 39073


# categorical data encoding. Our target: income--> binary
income = dataset[['income']]
ordinal_encoder = OrdinalEncoder()
income_encoded = ordinal_encoder.fit_transform(income)
dataset.income = income_encoded

workclass = dataset[['workclass']]
workclass_encoded = ordinal_encoder.fit_transform(workclass)
dataset.workclass = workclass_encoded

education = dataset[['education']]
education_encoded = ordinal_encoder.fit_transform(education)
dataset.education = education_encoded
