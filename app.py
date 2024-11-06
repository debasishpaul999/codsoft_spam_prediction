import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import pandas as pd, numpy as np

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl','rb'))
mnb = pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Classifier")

page_bg_img = """
<style>
[data-testid = "stAppViewContainer"]{
background: conic-gradient(from 90deg at 63% 50%, #FF0037, rgba(255,144,168,1))
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

input_text = st.text_area("Enter the message")

def text_transformer(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    x =  []
    for i in text:
        if i.isalnum():
            x.append(i)
    text = x[:]
    x.clear()
    for i in text:
        if i not in stopwords.words('english'):
            if i not in string.punctuation:
                x.append(i)

    text = x[:]
    x.clear()
    for i in text:
        x.append(ps.stem(i))

    return " ".join(x)

if st.button('Predict'):

    transformed_text = text_transformer(input_text)

    vector_input = tfidf.transform([transformed_text])

    result = mnb.predict(vector_input)[0]

    if result == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")
