from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load 20newsgroups dataset
train_data = fetch_20newsgroups(subset='train')
test_data = fetch_20newsgroups(subset='test')

# create a pipeline, it for text processing and logistic regression
pipeline = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=1000))

# fit the pipeline
pipeline.fit(train_data.data, train_data.target)

# predict the test data
# logic: put the test data into CountVectorizer, then put the result into LogisticRegression to predict the target
y_pred = pipeline.predict(test_data.data)

# print the accuracy
print("Accuracy: %.2f" % accuracy_score(test_data.target, y_pred))
