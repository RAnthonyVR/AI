# Ricardo Antonio Vázquez Rodríguez A01209245
# Intelligent systems
# Date: August 19 2021

# Gradient descent: optimizes the values for the slope and the Y intercept.
# It does more calculations while getting closer to the optimal solution. (big steps when it is far away, baby steps when close)
# Can be used when it isn´t possible to find the derivarative of an optimal solution (minimum error between predictions and real values)
# GRADIENT: two or more derivatives of the same function
# Used to descent to the lowest point in the loss function

# y = (slope)x + (yIntercept)

# Import libraries
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import linear_model

# Loss function: Sum of squared residuals
# sum = (y1 - (yIntercept + slope * x1))^2 + (y2 - (yIntercept + slope * x2))^2 + ... (until xn and yn)

# Partial derivative for loss function (chain rule). Gradient
# 	Partial derivatve of slope
		# -2(y1 - (yIntercept + slope * x1)) + -2(y2 - (yIntercept + slope * x2)) + ... (until xn and yn)
# 	Partial derivative of yIntercept
		# -2*x1*(y1 - (yIntercept + slope * x1)) + -2*x2*(y2 - (yIntercept + slope * x2)) + ... (until xn and yn)

def evaluateLinearFunction(coefficients, xs, yIntercept):
	'''
	Function to predict the value of y given coefficients and x value(s)

	Args:
		coefficients (list): slopes of the function
		xs (list): independent terms to predict
		yIntercept (float): bias of the model
	Returns:
		predictedY (float): predicted result (Y) 
	'''

	predictedY = 0

	numberOfTerms = len(coefficients)

	for termNumber in range(numberOfTerms):
		predictedY += coefficients[termNumber] * xs[termNumber]

	predictedY += yIntercept

	return predictedY


def evaluatePartialDerivativeSlope(yIntercept, coefficients, xs, ys, coefficientNumber):
	'''
		Function that evaluates the partial derivative of the loss function with respect to the slope.
	'''

	resultSlope = 0
	# 	Partial derivative of slope
		# -2*x1*(y1 - (yIntercept + slope * x1)) + -2*x2*(y2 - (yIntercept + slope * x2)) + ... (until xn and yn)
	for x, y in zip(xs,ys):
		resultSlope += (x[coefficientNumber])*(evaluateLinearFunction(coefficients, x, yIntercept) - y) # *-2

	# get average
	resultSlope = resultSlope / len(xs)

	return resultSlope


def evaluatePartialDerivativeYIntercept(yIntercept, coefficients, xs, ys):
	resultSlope = 0
	# 	Partial derivatve of YIntercept
		# -2(y1 - (yIntercept + slope * x1)) + -2(y2 - (yIntercept + slope * x2)) + ... (until xn and yn)
	for x, y in zip(xs,ys):
		resultSlope += (evaluateLinearFunction(coefficients, x, yIntercept) - y) # *-2

	# get average
	resultSlope = resultSlope / len(xs)

	return resultSlope

def getCoefficientsWithGradientDescent(slopesCoefficients, yIntercept, learningRate, currentNumberOfSteps, maximumNumberOfSteps, xs ,y):

	stepSize_YIntercept = 1
	stepSize_XCoefficientSlope = [] # one per additional coefficient (list)
	for _ in range(len(slopesCoefficients)):
		stepSize_XCoefficientSlope.append(1)

	while (currentNumberOfSteps < maximumNumberOfSteps): #or ((stepSize_YIntercept > 0.001) and (stepSize_Slope > 0.001)):
		# get slope of loss function per coefficient
		stepSize_YIntercept = evaluatePartialDerivativeYIntercept(yIntercept, slopesCoefficients, xs, y) * learningRate
		for coefficientNumber in range(len(slopesCoefficients)):
			stepSize_XCoefficientSlope[coefficientNumber] = evaluatePartialDerivativeSlope(yIntercept, slopesCoefficients, xs, y, coefficientNumber) * learningRate

		# Update coefficients
		yIntercept = yIntercept - stepSize_YIntercept

		for coefficientNumber in range(len(slopesCoefficients)):
			slopesCoefficients[coefficientNumber] -= stepSize_XCoefficientSlope[coefficientNumber]

		# print("slope: ", slope)
		# print("Y intercept: ", yIntercept) 
		# print("") 

		currentNumberOfSteps += 1

	return slopesCoefficients, yIntercept

def makeQueries(coefficients, yIntercept):
	'''
	Function that handles inputs by the user to predict

	Args:
		coefficients (list): slopes of the function
		yIntercept (float): bias of the model
	Returns:
		None: Prints the predicted result (Y) 
	'''

	numberOfVariables = len(coefficients)

	xs = []
	for _ in range(len(coefficients)):
		try:
			x = int(input("Insert an x value "))
		except:
			print("Non numeric value was entered!")

		xs.append(x)

	# predict result
	Yprediction = evaluateLinearFunction(coefficients, xs, yIntercept)

	# print predicted result
	print("Predicted result: ", Yprediction)


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

def calculateMSE(calculatedResults, realResults):
	mse = 0
	sumSquares = 0
	length = 0
	for forecast, actual in zip(calculatedResults, realResults):
		sumSquares += (actual - forecast) ** 2
		length += 1

	mse = sumSquares / length

	return mse


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
		sumSquares += (actual - forecast) ** 2
		squaredFromMeanY += (actual - meanYs) ** 2
		length += 1

	r2 = (sumSquares / squaredFromMeanY)

	return r2

def main():

	# Initial values
	slopesCoefficients = []
	yIntercept = 0
	learningRate = 0.01
	currentNumberOfSteps = 0
	maximumNumberOfSteps = 150

	# dummy data (single variable)
	# x = [1,2,3,4,5]
	# y = [2,4,6,8,10]

	# dummy data (multi variable)
	# x = [[1,1],[2,2],[3,3],[4,4],[5,5],[2,2],[3,3],[4,4]]
	# y = [2,4,6,8,10,2,5.5,16]

	# # Import the dataset
	dataset = pd.read_csv('student-mat.csv', sep = ';')
	# # Load the diabetes dataset thta are already cleaned and preprocessed
	x = list(dataset.iloc[:, 30].values) # rows and columns
	#x = list(dataset['G1', 'G2'])# rows and columns
	y = list(dataset.iloc[:, 32].values)

	# initial value (0) of the coefficients depending on the number of variables
	try: # multiple variables coefficient
		for _ in range(len(x[0])):
			slopesCoefficients.append(0)
	except: # one dimentional coefficient
		slopesCoefficients = [0] # slope m (y = mx + b)
		# make each element a list as  in multivariable example
		formatedX = []
		for xValue in x:
			formatedX.append(list(x)) 
		x = formatedX

	# TO DO: Scale data for convergence

	# Divide into train and test
	xTraining, yTraining, xTesting, yTesting =  divideTrainAndTestData(x, y)

	# Trainning
	coefficients, yIntercept = getCoefficientsWithGradientDescent(slopesCoefficients, yIntercept, learningRate, currentNumberOfSteps, maximumNumberOfSteps, x, y)

	print("Final slope: ", coefficients)
	print("Final y intercept: ", yIntercept)

	# Train error
	lasso = linear_model.Lasso()
	crossvalidation_errors = cross_val_score(lasso, xTraining, yTraining)
	trainError = 0
	# average
	for error in crossvalidation_errors:
		trainError += error

	trainError = trainError / len(crossvalidation_errors)
	print("Train error: ", trainError)

	# Validation error ?
	print("Validation error: ")

	# TO DO: Additional testing for production

	testResultsY = [evaluateLinearFunction(coefficients, x, yIntercept) for x in xTesting]

	# Get Test error
	print("Test error: ")
	print("MSE: ", calculateMSE(testResultsY, yTesting))
	# print("R2Score: ", calculateR2Score(testResultsY, yTesting))

	# Query prediction
	makeQueries(coefficients, yIntercept)


if __name__ == "__main__":
    main()










