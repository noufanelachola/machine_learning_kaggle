import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def getMae(maxNode):
    model = DecisionTreeRegressor(max_leaf_nodes=maxNode,random_state=1)
    model.fit(trainX,trainy)

    predictions = model.predict(valX)

    print("Mean abosulte error for "+str(maxNode)+" : "+str(mean_absolute_error(valy,predictions)))


data = pd.read_csv("melb_data.csv")
y = data.Price

data_features = ["Rooms","Bathroom","Landsize","Lattitude","Longtitude"]
X = data[data_features]

trainX,valX,trainy,valy = train_test_split(X,y,random_state=1)

nodes = [5,50,500,5000]

for node in nodes:
    getMae(node)




