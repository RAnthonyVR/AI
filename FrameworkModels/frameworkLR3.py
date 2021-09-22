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

def calculateR2Score(calculatedResults, realResults):
	sumSquares = 0
	length = 0
	squaredFromMeanY = 0
	meanYs = 0
	# calculate average real y
	for y in realResults:
		meanYs += y
		length += 1

	meanYs = meanYs / length

	length = 0

	for forecast, actual in zip(calculatedResults, realResults):
		sumSquares += (forecast - meanYs) ** 2
		squaredFromMeanY += (actual - meanYs) ** 2
		length += 1

	r2 = (sumSquares / squaredFromMeanY)

	return r2

dataset = pd.read_csv('../student-mat.csv', sep = ';')
dataset = pd.get_dummies(dataset, columns=["school"])
x = dataset[["Medu", "Fedu", "studytime", "absences", "G1", "G2", "school_GP", "school_MS"]]

y = list(dataset.iloc[:, 31].values)

xTraining, xTesting, yTraining, yTesting = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)
xTraining, yTraining, xTesting, yTesting =  divideTrainAndTestData(x, y)

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

print("R2Score: ", calculateR2Score(predictions, yTesting))


