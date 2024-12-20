import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

model = DecisionTreeRegressor(random_state=1)
data = pd.read_csv("melb_data.csv")
imputer = SimpleImputer()

y = data.Price

melb_features = data.drop(["Price"],axis=1)
X = melb_features.select_dtypes(exclude=["object"])

train_X, val_X, train_y, val_y = train_test_split(X,y,train_size=0.8,test_size=0.2)

imputed_train_X = pd.DataFrame(imputer.fit_transform(train_X))
imputed_val_X = pd.DataFrame(imputer.transform(val_X))

imputed_train_X.columns = train_X.columns
imputed_val_X.columns = val_X.columns

model.fit(imputed_train_X,train_y)

predictions = model.predict(imputed_val_X)

print(f"The MAE is : {mean_absolute_error(val_y,predictions)}")

