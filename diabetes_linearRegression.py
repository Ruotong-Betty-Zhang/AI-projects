from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# load the datasets
diabetes = datasets.load_diabetes()
x = diabetes.data
y = diabetes.target

# split the datasets, 20% for testing, 80% for training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 0)

# create the model
lr = LinearRegression()

# train the model
lr.fit(x_train, y_train)

# test the model
y_pred = lr.predict(x_test)

# print the mean squared error
# the mean squared error more close to 0, the better the model
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

# print the accuracy of the model
# the accuracy more close to 1, the better the model
print("Accuracy: %.2f" % lr.score(x_test, y_test))