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







