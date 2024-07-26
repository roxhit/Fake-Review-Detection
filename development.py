#LIBRARIES
import streamlit as st
import pickle
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

#LOAD PICKLE FILES
model = pickle.load(open('pickle/best_model.pkl', 'rb'))
vectorizer = pickle.load(open('pickle/count_vectorizer.pkl', 'rb'))

#FOR STREAMLIT
nltk.download('stopwords')

#TEXT PREPROCESSING
sw = set(stopwords.words('english'))
def text_preprocessing(text):
    txt = TextBlob(text)
    result = txt.correct()
    removed_special_characters = re.sub("[^a-zA-Z]", " ", str(result))
    tokens = removed_special_characters.lower().split()
    stemmer = PorterStemmer()

    cleaned = []
    stemmed = []

    for token in tokens:
        if token not in sw:
            cleaned.append(token)

    for token in cleaned:
        token = stemmer.stem(token)
        stemmed.append(token)

    return " ".join(stemmed)

#TEXT CLASSIFICATION
def text_classification(text):
    if len(text.strip()) < 1:
        st.warning("Please enter some text.")
    else:
        with st.spinner("Classification in progress..."):
            cleaned_review = text_preprocessing(text)
            process = vectorizer.transform([cleaned_review]).toarray()
            prediction = model.predict(process)
            p = ''.join(str(i) for i in prediction)

            if p == 'True':
                st.success("The review entered is Legitimate.")
            if p == 'False':
                st.error("The review entered is Fraudulent.")

#PAGE FORMATTING AND APPLICATION
def main():
    # Custom CSS
    st.markdown("""
        <style>
            body {
                background-color: #f5f5f5;
                color: #333;
            }
            .stButton button {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                transition-duration: 0.4s;
            }
            .stButton button:hover {
                background-color: white;
                color: black;
                border: 2px solid #4CAF50;
            }
            .stExpander {
                border: 2px solid #4CAF50;
                border-radius: 5px;
                padding: 10px;
            }
            .stTextArea textarea {
                border: 2px solid #4CAF50;
                border-radius: 5px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                document.querySelector('textarea').placeholder = 'Enter your review here...';
            });
        </script>
    """, unsafe_allow_html=True)

    st.title("Fraud Detection in Online Consumer Reviews")

    # --EXPANDERS--
    abstract = st.expander("Abstract")
    if abstract:
        abstract.write("In today's world, both businesses and customers believe reviews to be quite beneficial. It's no surprise that review fraud has devalued the whole experience, from nasty reviews putting harm to the business's credibility to breaking international laws. This has been seen as a developing problem, and because it is related to natural language processing, it was critical to develop various machine learning methodologies and techniques to achieve a breakthrough in this sector. Many e-commerce sites, such as Amazon, have their systems in place, including Verified Purchase, which labels review language as accurate when items are purchased directly from the website. This work proposes to use Amazon's verified purchases label to train three classifiers for supervised training on Amazon’s labelled dataset. MNB, SVM, and LR were chosen as classifiers, and model tuning was done using two distinct vectorizers, Count Vectorizer and TF-IDF Vectorizers. Overall, all of the trained models had an accuracy rate of 80%, indicating that the vectorizers functioned admirably and that there are distinctions between false and actual reviews. Out of the two, the count vectorizer improved the models' performance more, and out of the three inside counts, LR performed the best, with an accuracy rate of 85% and a recall rate of 92%. The LR classifier was used, and it was accessible to the public to see if the reviews entered were genuine or not, with a probability score.")
    
    # --CHECKBOXES--
    st.subheader("Information on the Classifier")
    if st.checkbox("About Classifer"):
        st.markdown('**Model:** Logistic Regression')
        st.markdown('**Vectorizer:** Count')
        st.markdown('**Test-Train splitting:** 40% - 60%')
        st.markdown('**Spelling Correction Library:** TextBlob')
        st.markdown('**Stemmer:** PorterStemmer')

    if st.checkbox("Evaluation Results"):
        st.markdown('**Accuracy:** 85%')
        st.markdown('**Precision:** 80%')
        st.markdown('**Recall:** 92%')
        st.markdown('**F-1 Score:** 85%')

    #--IMPLEMENTATION OF THE CLASSIFIER--
    st.subheader("Fake Review Classifier")
    review = st.text_area("Enter Review: ")
    if st.button("Check"):
        text_classification(review)

#RUN MAIN
main()
