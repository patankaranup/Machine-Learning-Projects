import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text) # returns a list of words from sentences
    new_text = []
    for word in text:
        if word.isalnum(): # removes special characters 
            new_text.append(word)
            
    text = new_text[:]
    new_text.clear()
    
    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            new_text.append(word)
            
    text = new_text[:]
    new_text.clear()
    
    for word in text:
        new_text.append(ps.stem(word))
    
    return " ".join(new_text)



tfidf = pickle.load(open('./vectorizer.pkl','rb'))
model = pickle.load(open('./model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the  message")

if st.button('Predict'):

    # Preprocess 
    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict 
    result = model.predict(vector_input)[0]

    # Display

    if(result == 1):
        st.header("Spam")
    else :
        st.header("Not Spam")