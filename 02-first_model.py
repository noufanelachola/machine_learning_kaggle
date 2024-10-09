import pandas as pd

data = pd.read_csv("melb_data.csv")
y = data.Price

data_features = ["Rooms","Bathroom","Landsize","Lattitude","Longtitude"]
X = data[data_features]



print(X.head())