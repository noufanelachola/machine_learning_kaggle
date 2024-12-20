import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("./melb_data.csv")

model = RandomForestRegressor(random_state=1)

y = data.Price
X = data.drop("Price",axis=1)

X_train,X_val,y_train,y_val = train_test_split(X,y,random_state=0)

cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
X_train.drop(cols_with_missing,axis=1,inplace=True)
X_val.drop(cols_with_missing,axis=1,inplace=True)

low_cardinity_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 
                        and X_train[cname].dtype == "object"]

numerical_cols = [cname for cname in X_train if X_train[cname].dtype in ["int64","float64"] ]

my_cols = low_cardinity_cols + numerical_cols

x_train = X_train[my_cols].copy()
x_val = X_val[my_cols].copy()


print(f"low cardintity : {low_cardinity_cols}")
print(f"numerial cols : {numerical_cols}")

print(x_train.head())


# model.fit(X_train,y_train)
# prediction = model.predict(X_val)

# print(f"MEAN Absolute Error : {mean_absolute_error(y_val,prediction)}")



