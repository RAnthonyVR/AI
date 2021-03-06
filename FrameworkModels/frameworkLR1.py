import sklearn
import pandas as pd
import sklearn.model_selection
import sklearn.linear_model

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


dataset = pd.read_csv('../student-mat.csv', sep = ';')

x = list(dataset.iloc[:, 30:32].values) # rows and columns

y = list(dataset.iloc[:, 32].values)

xTraining, xTesting, yTraining, yTesting = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)

hypothesis = sklearn.linear_model.LinearRegression()

hypothesis.fit(xTraining, yTraining)
score = hypothesis.score(xTesting, yTesting)
print(score)

print('coefficients: ', hypothesis.coef_)
print('Y intercept: ', hypothesis.intercept_)


# prediction 

# print('prediction: with 10 and 5: ', hypothesis.predict([[10, 5]]))
# hypothesis.predict([[10, 5]])


# Divide into train and test
xTraining, yTraining, xTesting, yTesting =  divideTrainAndTestData(x, y)
predictions = hypothesis.predict(xTesting)
predictions = predictions.tolist()


print("MSE:")
acum = 0
for index, prediction in enumerate(predictions):
	acum += (yTesting[index] - prediction) ** 2

print(acum / len(predictions))

print("r2 score: ", hypothesis.score(xTesting, yTesting))
