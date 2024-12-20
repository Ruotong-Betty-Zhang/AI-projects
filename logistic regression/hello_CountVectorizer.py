from sklearn.feature_extraction.text import CountVectorizer

text = [
    "I love programming", 
    "I am a student, and I love programming, I am learning python and machine learning", 
    "I love AI"
]

# create a CountVectorizer
vectorizer = CountVectorizer()

# fit the vectorizer
X = vectorizer.fit_transform(text)
print(X.toarray())

# the output is a sparse matrix
# the number of rows is the number of rows means the number of words
# 0,1,2 means how many times the word appears in this row


