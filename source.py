import matplotlib.pyplot as plt
import math

clean_data = []

def sigma(func):
	data = clean_data
	res = 0
	for x in data:
		res+= func(x)
	return res


def exponentialFit():
	# y = a(e ** (bx))
	# log(y) = log(a) + bxlog(e)
	# Y = A + Bx
	# Y = log(y) # A = log(a) # B = blog(e)
	
	# sigma(Y) = nA + Bsigma(x)            c1 = A(x1) + B(y1)
	# sigma(xY) = Asigma(x) + Bsigma(x**2) c2 = A(x2) + B(y2)
	
	# a = antilig(A) b = B/(log(e))

	c1 = sigma(lambda x : math.log(x[1], 10))
	x1 = len(clean_data)
	y1 = sigma(lambda x : x[0])

	c2 = sigma(lambda x : ((x[0])*(math.log(x[1],10))))
	x2 = y1
	y2 = sigma(lambda x : x[0] ** 2)


	import numpy as np
	A = np.array([[x1, y1], [x2, y2]])
	B = np.array([c1, c2])
	X = np.linalg.solve(A, B)
	A = X[0]
	B = X[1]

	a = 10 ** A
	b = B*(math.log(10))

	return (a,b)


def straight_line_fit():
	# ax + b = y            # x1 = sigma(x[i] ** 2)    	# x2 = sigma(x[i])
	# a(x1) + b(y1) = c1    # y1 = sigma(x[i])         	# y2 = n
	# a(x2) + b(y2) = c2    # c1 = sigma(x[i]y[i])     	# c2 = sigma(y[i])
	
	data = clean_data
	if data == []:
		print("First Load and then try!")
		return 0

	x1 = sigma(lambda x : x[0] **2 )
	y1 = sigma(lambda x : x[0])
	c1 = sigma(lambda x : x[0] * x[1])

	x2 = y1
	y2 = len(data)
	c2 = sigma(lambda x : x[1])

	# AX = B
	import numpy as np
	A = np.array([[x1, y1], [x2, y2]])
	B = np.array([c1, c2])
	X = np.linalg.solve(A, B)
	return tuple(X)


def parabola_fit():

	# y = a + bx + c(x**2)          # c1 = sigma(y[i])	     	# c2 = sigma(x[i]*y[i])     # c3 = sigma((x[i]** 2)*y[i])
	# c1 = a(x1) + b(y1) + c(z1) 	# x1 = n	             	# x2 = sigma(x[i])       	# x3 = sigma(x[i] ** 2)
	# c2 = a(x2) + b(y2) + c(z2) 	# y1 = sigma(x[i])	     	# y2 = sigma(x[i] ** 2)  	# y3 = sigma(x[i] ** 3)
	# c3 = a(x3) + b(y3) + c(z3) 	# z1 = sigma(x[i] ** 2)	 	# z2 = sigma(x[i] ** 3)  	# z3 = sigma(x[i] ** 4)
	
	data = clean_data
	
	if data == []:
		print("First Load and then try!")
		return 0

	c1 = sigma(lambda x : x[1])
	x1 = len(data)
	y1 = sigma(lambda x : x[0])
	z1 = sigma(lambda x : x[0] ** 2)

	c2 = sigma(lambda x : x[0] * x[1])
	x2 = sigma(lambda x : x[0])
	y2 = sigma(lambda x : x[0] ** 2)
	z2 = sigma(lambda x : x[0] ** 3)

	c3 = sigma(lambda x : (x[0]**2) * x[1])
	x3 = sigma(lambda x : x[0] ** 2)
	y3 = sigma(lambda x : x[0] ** 3)
	z3 = sigma(lambda x : x[0] ** 4)

	# AX = B
	import numpy as np
	A = np.array([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
	B = np.array([c1, c2, c3])
	X = np.linalg.solve(A, B)
	return tuple(X)


def getEquation():
	print("Loading ........")
	x1 = straight_line_fit()
	x2 = parabola_fit()
	x3 = exponentialFit()
	print("Y = ({:.4f}) + ({:.4f})X".format(x1[0],x1[1]))
	print("Y = ({:.4f}) + ({:.4f})X + ({:.4f})X\u00b2".format(x2[0],x2[1],x2[2]))
	print("Y = ({:.4f}) x e^({:.4f}X) ".format(x3[0],x3[1]))


def calculateYOnLine(coefficients,x):
	return (coefficients[0])*x + (coefficients[1])


def calculateYOnParabola(coefficients,x):
	return coefficients[0] + coefficients[1]*x + coefficients[2]*x**2


def calculateYOnExponentialCurve(coefficients,x):
	return coefficients[0]*(math.exp(coefficients[1]*x))


def findPointsOnLine(coefficients,min,max):
	a = []
	b = []

	for x in range(min,max+1,1):
		y = calculateYOnLine(coefficients,x)
		a.append(x)
		b.append(y)

	return (a,b)


def findPointsOnParabola(coefficients,min,max):
	a = []
	b = []

	for x in range(min,max+1,1):
		y = calculateYOnParabola(coefficients,x)
		a.append(x)
		b.append(y)

	return (a,b)


def findPointsOnExponentialCurve(coefficients,min,max):
	a = []
	b = []

	for x in range(min,max+1,1):
		y = calculateYOnExponentialCurve(coefficients,x)
		a.append(x)
		b.append(y)

	return (a,b)


def plotGraph(coefficients,func,axes):
	# func = findPointsOnParabola or findPointsOnLine
	input = clean_data

	if input == []:
		print("First Load and then try!")
		return 0
	
	xCord = []
	yCord = []

	for a2 in input:
		xCord.append(a2[0])
		yCord.append(a2[1])

	points = func(coefficients,0,int(max(xCord)))
	a = points[0]
	b = points[1]

	axes.plot(a,b)
	axes.plot(xCord,yCord,'y.')
	return axes


def predictValueOnLine(x, coefficients,axes):
	y = calculateYOnLine(coefficients,x)
	axes.plot(x,y,'r^')
	axes.plot([0,x],[y,y],'g--')
	axes.plot([x,x],[0,y],'g--')
	return y


def predictValueOnParabola(x,coefficients,axes):
	# x is input, y is output
	y = calculateYOnParabola(coefficients,x)
	axes.plot(x,y,'r^')
	axes.plot([0,x],[y,y],'g--')
	axes.plot([x,x],[0,y],'g--')
	return y


def predictValueOnExponentialCurve(x,coefficients,axes):
	y = calculateYOnExponentialCurve(coefficients,x)
	axes.plot(x,y,'r^')
	axes.plot([0,x],[y,y],'g--')
	axes.plot([x,x],[0,y],'g--')
	return y


def predict():
	print("1. Line\n2. Parabola\n3. Exponential Curve\n4. All")
	userChoice = int(input("Option : "))
	fig= plt.figure()
	axes=fig.add_subplot(111)
	if userChoice == 1:
		x = int(input("x : "))
		coefficients = straight_line_fit()
		plotGraph(coefficients, findPointsOnLine,axes)
		print("Predicted Value(By Line Fit) : ", predictValueOnLine(x, coefficients, axes))
		plt.show()
	elif userChoice == 2:
		x = int(input("x : "))
		coefficients = parabola_fit()
		plotGraph(coefficients,findPointsOnParabola,axes)
		print("Predicted Value(By Parabolic Fit) : ",predictValueOnParabola(x,coefficients,axes))
		plt.show()
	elif userChoice == 3:
		x = int(input("x : "))
		coefficients = exponentialFit()
		plotGraph(coefficients, findPointsOnExponentialCurve,axes)
		print("Predicted Value (By Exponential Fit): ", predictValueOnExponentialCurve(x,coefficients,axes))
		plt.show()
	elif userChoice == 4:
		x = int(input("x : "))
		coefficients = straight_line_fit()
		plotGraph(coefficients, findPointsOnLine,axes)
		print("Predicted Value(By Line Fit) : ", predictValueOnLine(x, coefficients, axes))
		coefficients = parabola_fit()
		plotGraph(coefficients,findPointsOnParabola,axes)
		print("Predicted Value(By Parabolic Fit) : ",predictValueOnParabola(x,coefficients,axes))
		coefficients = exponentialFit()
		plotGraph(coefficients, findPointsOnExponentialCurve,axes)
		print("Predicted Value (By Exponential Fit): ", predictValueOnExponentialCurve(x,coefficients,axes))
		plt.show()
	else:
		print("Wrong Input! Try again!")
		return 0


def plotGraph2():
	# internally calls plotGraph

	print("1. Line\n2. Parabola\n3. Exponential Curve\n4. All")
	userChoice = int(input("Option : "))
	fig= plt.figure()
	axes=fig.add_subplot(111)

	if userChoice == 1:
		coefficients = straight_line_fit()
		plotGraph(coefficients,findPointsOnLine,axes)
		plt.show()
	elif userChoice == 2:
		coefficients = parabola_fit()
		plotGraph(coefficients,findPointsOnParabola,axes)
		plt.show()
	elif userChoice == 3:
		coefficients = exponentialFit()
		plotGraph(coefficients,findPointsOnExponentialCurve,axes)
		plt.show()
	elif userChoice == 4:
		coefficients = straight_line_fit()
		plotGraph(coefficients,findPointsOnLine,axes)
		coefficients = parabola_fit()
		plotGraph(coefficients,findPointsOnParabola,axes)
		coefficients = exponentialFit()
		plotGraph(coefficients,findPointsOnExponentialCurve,axes)
		plt.show()

	else:
		print("Wrong Input! Try Again!")
		return 0


def load():
	print("FILE NAME : ", end = "")
	fileName = input()
	
	try:
		file = open(fileName,'r',encoding="utf8")
	except FileNotFoundError as e:
		print("No such file or directory: ", '"', fileName, '"')
		return 0
	
	print("Loading ........")
	rawData = file.read().splitlines()

	columnNames = rawData[0].split(',')
	# print(columnNames)
	rowData = rawData[1].split(",")
	count = 1
	columnNumberMapping = {}
	print(rowData)
	for i in range(0,len(columnNames)):
		try:
			float(rowData[i])
			columnNumberMapping[count] = (columnNames[i],i)
			print(count," : ",columnNumberMapping[count][0])
			count += 1
		except :
			continue
	# print(columnNames)
	# print(columnNumberMapping)

	print("\nChoose an indepedent")
	x = int(input("X : "))

	if x not in columnNumberMapping.keys():
		print("Wrong Input")
		return

	print("\nChoose a dependent")
	y = int(input("Y : "))

	if y not in columnNumberMapping.keys():
		print("Wrong Input")
		return

	if x == y :
		print("Wrong Input!")
		return

	for i in range(1,len(rawData)):
		rowData = rawData[i].split(",")
		try:
			clean_data.append((int(float(rowData[columnNumberMapping[x][1]])), float(rowData[columnNumberMapping[y][1]])))
		except:
			continue
	# print(clean_data)
	# print(len(clean_data), "lines loaded.")


def RSSError(coefficients,func):
	# func = calculateYOnLine or calculateYOnParabola
	input = clean_data
	sum  = 0
	for a in input:
		sum+=(a[1]-func(coefficients,a[0]))**2
	return sum/len(input)


def maxError(coefficients,func):
	# func = calculateYOnLine or calculateYOnParabola
	input = clean_data
	a = 0
	for x in input:
		b = func(coefficients,x[0])-x[1]
		a = max(a,abs(b))
	return a


def avgError(coefficients,func):
	# func = calculateYOnLine or calculateYOnParabola
	input = clean_data
	sum = 0
	for x in input:
		sum+=abs(func(coefficients,x[0])-x[1])
	return sum/len(input)


def rss():

	if clean_data == []:
		print("First Load and then try!")
		return 0
	
	print("1. Line\n2. Parabola\n3. Exponential Curve\n4. All")
	x = int(input("Option : "))

	if x == 1:
		coefficients = straight_line_fit()
		print("Line Fit : \n")
		print("RSS Error : ",RSSError(coefficients,calculateYOnLine))
		print("AVG Error : ",avgError(coefficients,calculateYOnLine))
		print("MAX Error : ",maxError(coefficients,calculateYOnLine))
	elif x == 2:
		coefficients = parabola_fit()
		print("\nParabolic Fit: \n")
		print("RSS Error : ",RSSError(coefficients,calculateYOnParabola))
		print("AVG Error : ",avgError(coefficients,calculateYOnParabola))
		print("MAX Error : ",maxError(coefficients,calculateYOnParabola))
	elif x == 3:
		coefficients = exponentialFit()
		print("\nExponential Fit: \n")
		print("RSS Error : ",RSSError(coefficients,calculateYOnExponentialCurve))
		print("AVG Error : ",avgError(coefficients,calculateYOnExponentialCurve))
		print("MAX Error : ",maxError(coefficients,calculateYOnExponentialCurve))
	elif x == 4:
		coefficients = straight_line_fit()
		print("Line Fit : \n")
		print("RSS Error : ",RSSError(coefficients,calculateYOnLine))
		print("AVG Error : ",avgError(coefficients,calculateYOnLine))
		print("MAX Error : ",maxError(coefficients,calculateYOnLine))
		coefficients = parabola_fit()
		print("\nParabolic Fit: \n")
		print("RSS Error : ",RSSError(coefficients,calculateYOnParabola))
		print("AVG Error : ",avgError(coefficients,calculateYOnParabola))
		print("MAX Error : ",maxError(coefficients,calculateYOnParabola))
		coefficients = exponentialFit()
		print("\nExponential Fit: \n")
		print("RSS Error : ",RSSError(coefficients,calculateYOnExponentialCurve))
		print("AVG Error : ",avgError(coefficients,calculateYOnExponentialCurve))
		print("MAX Error : ",maxError(coefficients,calculateYOnExponentialCurve))
	else:
		print("Wrong Input! Try again!")


def menu():
	import platform
	import os
	
	operatingSystem = platform.system()

	if operatingSystem == "Windows":
		os.system('cls')
	else:
		os.system('clear')


	while True:
		print("=" * 50)
		print("1. LOAD")
		print("2. GET EQUATION")
		print("3. PREDICT")
		print("4. RESIDUAL ERROR")
		print("5. PLOT")
		print("6. EXIT")
		print("=" * 50)
		option = int(input("OPTION : "))
		print("=" * 50)
		if option == 1:
			load()
		elif option == 2:
			getEquation()
		elif option == 3:
			predict()
		elif option == 4:
			rss()
		elif option == 5:
			plotGraph2()
		elif option == 6:
			return 0
		else:
			continue


def main():
	menu()

if __name__ == '__main__':
	menu()