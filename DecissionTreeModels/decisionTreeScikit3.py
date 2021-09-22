# Decision tree regression

# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

from sklearn.tree import DecisionTreeRegressor
import matplotlib as mp
import pandas as pd
from sklearn import tree

def divideTrainAndTestData(xs, ys):
	'''
	80% train data, 20 % test data 
	'''
	numerOfTotalRows = len(ys)

	# Get 80%
	numerOfRowsTraining = int((numerOfTotalRows * 80) / 100)

	# Assign 80%
	xTraining = xs[:numerOfRowsTraining]
	yTraining = ys[:numerOfRowsTraining]

	# Assign remaining 20%
	xTesting = xs[numerOfRowsTraining:]
	yTesting = ys[numerOfRowsTraining:]

	return xTraining, yTraining, xTesting, yTesting

# Load dataset
dataset = pd.read_csv('../student-mat.csv', sep = ';')
# dependent and dependent variables
x = dataset[["Medu", "Fedu", "studytime", "absences", "G1", "G2"]]
y = list(dataset.iloc[:, 32].values)



# split train and test
xTraining, yTraining, xTesting, yTesting = divideTrainAndTestData(x, y)

# Get model object from sckit
hypothesis = DecisionTreeRegressor(random_state = 0)

# Train model
hypothesis.fit(xTraining, yTraining)

tree.plot_tree(hypothesis)
plt.show()

# make predictions
predictions = hypothesis.predict(xTesting)
predictions = predictions.tolist()

# get MSE
print("MSE:")
acum = 0
for index, prediction in enumerate(predictions):
	acum += (yTesting[index] - prediction) ** 2

print(acum / len(predictions))



