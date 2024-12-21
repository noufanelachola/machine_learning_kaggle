import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def score_dataset(X_train,X_val,y_train,y_val):
    model = RandomForestRegressor(random_state=1)
    model.fit(X_train,y_train)
    prediction = model.predict(X_val)
    return mean_absolute_error(y_val,prediction)

data = pd.read_csv("./melb_data.csv")

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

s = x_train.dtypes == "object"
object_cols = list(s[s].index)

# Approach 1 (Drop column)
drop_x_train = x_train.select_dtypes(exclude=["object"])
drop_x_valid = x_val.select_dtypes(exclude=["object"])

print(f"MAE approach 1 : {score_dataset(drop_x_train,drop_x_valid,y_train,y_val)}")


# Approach 2 (Ordinal Encoding)
from sklearn.preprocessing import OrdinalEncoder

label_x_train = x_train.copy()
label_x_valid = x_val.copy()

ordinalEncoder = OrdinalEncoder()

label_x_train[object_cols] = ordinalEncoder.fit_transform(x_train[object_cols])
label_x_valid[object_cols] = ordinalEncoder.transform(x_val[object_cols])


print(f"MAE approach 2 : {score_dataset(label_x_train,label_x_valid,y_train,y_val)}")








