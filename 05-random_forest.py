import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("melb_data.csv")
data_features = ["Rooms","Bathroom","Landsize","Lattitude","Longtitude"]

y = data.Price
X = data[data_features]

train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=1)

decision_tree_model = DecisionTreeRegressor(random_state=1)
decision_tree_model.fit(train_X,train_y)

random_forest_model = RandomForestRegressor(random_state=1)
random_forest_model.fit(train_X,train_y)


prediction_decision = decision_tree_model.predict(val_X)
prediction_random = random_forest_model.predict(val_X)

mae_decision = mean_absolute_error(val_y,prediction_decision)
mae_random = mean_absolute_error(val_y,prediction_random)

print(f"Mean absolute error for decision tree is : {mae_decision}\n"+
      f"Mean absolute error for random forest is : {mae_random}")