from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load the MNIST dataset 数字识别
mnist = fetch_openml('mnist_784')

# data preprocessing
scaler = StandardScaler()
x = scaler.fit_transform(mnist.data)

# split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(x, mnist.target, test_size=0.2, random_state=42)

# create a logistic regression model
model = LogisticRegression(max_iter=1000)

# fit the model
model.fit(X_train, y_train)

# predict the test data
y_pred = model.predict(X_test)

# print the accuracy
print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))

