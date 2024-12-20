from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load the iris datasets
iris = datasets.load_iris()


x = iris.data
y = iris.target

# split the datasets, 20% for testing, 80% for training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 0, stratify=y)

# create the model, Logistic Regression will use the data set to predict is binary or milti class
# LogisticRegression(multi_class='ovr') means multi binary class 多个二分类
# LogisticRegression(multi_class='multinomial') means Softmax regression 做多分类
# set the max iteration: LogisticRegression(max_iter=1000)
lr = LogisticRegression()

# train the model
lr.fit(x_train, y_train)

# test the model
y_pred = lr.predict(x_test)

# print the accuracy of the model
# the accuracy more close to 1, the better the model
print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))

