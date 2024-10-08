import pandas as pd

home_data = pd.read_csv("melb_data.csv")

# print(home_data.head())
# print(home_data.shape)
# print(home_data.columns)
# print(home_data.isnull().sum())
print(home_data.describe())