import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

data = pd.read_csv("melb_data.csv")
y = data.Price

data_features = ["Rooms","Bathroom","Landsize","Lattitude","Longtitude"]
X = data[data_features]

trainX,valX,trainy,valy = train_test_split(X,y,random_state=1)

model = DecisionTreeRegressor(random_state=1)
model.fit(trainX,trainy)

predictions = model.predict(valX)

print("Mean abosulte error : "+str(mean_absolute_error(valy,predictions)))



