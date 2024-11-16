import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("melb_data.csv")

y = data.Price

melb_features = data.drop(["Price"],axis=1)
X = melb_features.select_dtypes(exclude=["object"])

train_X, val_X, train_y, val_y = train_test_split(X,y,train_size=0.8,test_size=0.2)

cols_with_missing = [col for col in train_X.columns if train_X[col].isnull().any()]

reduced_train_X = train_X.drop(cols_with_missing,axis=1)
reduced_valid_X = val_X.drop(cols_with_missing,axis=1)

model = DecisionTreeRegressor(random_state=1)
model.fit(reduced_train_X,train_y)

predictions = model.predict(reduced_valid_X)

print(f"The MAE is : {mean_absolute_error(val_y,predictions)}")

