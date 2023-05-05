import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pickle
import streamlit as st

review = pd.read_csv("review.csv")
review = review.rename(columns = {'text': 'review'}, inplace = False)
X = review.review
y = review.polarity
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, random_state = 1)

vector = CountVectorizer(stop_words = 'english',lowercase=False)
# fit the vectorizer on the training data
vector.fit(X_train)
X_transformed = vector.transform(X_train)
X_transformed.toarray()
# for test data
X_test_transformed = vector.transform(X_test)
naivebayes = MultinomialNB()
naivebayes.fit(X_transformed, y_train)

#save the model
saved_model = pickle.dumps(naivebayes)
#load saved model
s = pickle.loads(saved_model)
# Get user input and make a prediction

st.header('Santiment Classifier')
input = st.text_area("Review", value="")
if st.button("Classify"):
    vec = vector.transform([input]).toarray()
    pred = s.predict(vec)[0]
    cat ={0: 'NEGATIVE',1:'POSITIVE'}
    result=cat[pred]
    st.write("Sentiment classify:",result)