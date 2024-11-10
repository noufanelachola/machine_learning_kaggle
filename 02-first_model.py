import pandas as pd
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("melb_data.csv")
y = data.Price

data_features = ["Rooms","Bathroom","Landsize","Lattitude","Longtitude"]
X = data[data_features]



# print(X.describe())

model = DecisionTreeRegressor(random_state=1)

model.fit(X,y)

print("Making predictins for : ")
print(X.head())
print("The predictions are : ")
print(model.predict(X.head()))
print(y.head())