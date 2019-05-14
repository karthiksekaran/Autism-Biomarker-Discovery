import deap
from sklearn.linear_model import LogisticRegression
import pandas as pd
import feature_selection_ga

dataset = pd.read_csv("Final-26415.csv").values
x = dataset[:,:-1]
y = dataset[:,-1]
print(x.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5)
print(x_train.shape)
print(y_train.shape)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
fsga = feature_selection_ga.FeatureSelectionGA(model,x_train,y_train)
pop = fsga.generate(100)
print(len(pop))