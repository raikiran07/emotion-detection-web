import pandas as pd 
import numpy as np 
import pickle
import sklearn
import streamlit as st
import seaborn as sns
import math
import matplotlib.pyplot as plt
from transformers import pipeline


# Importing dependencies from transformers
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer



# Load tokenizer 
# tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")



# Load model 
# model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")










st.set_page_config(page_title='DetectEmotion', initial_sidebar_state = 'auto')

filename = 'pipeline.pkl'
with open(filename, 'rb') as file:
    pipe_lr = pickle.load(file)


input = st.text_area("Enter your text")

option = st.selectbox(
    'Would you like to summarize your input text',
    ('No','Yes'))
max = 0
min = 0

    





if st.button("Analyse"):
    
    output = pipe_lr.predict([input])

    prob = pipe_lr.predict_proba([input])

    st.header("Dominant Emotion : {}".format(output[0]))

   
    emotion_prob = []

    sum = 0
    for i in prob[0]:
        i = i * 100
        emotion_prob.append(math.floor(i))
        sum += i

    

   
    
    fig = plt.figure(figsize=(7, 5))
    plt.xticks(np.arange(0,100,5))
    plt.xlabel("probability percentage %")
    plt.ylabel("Emotions")

    d = {'Emotions':pipe_lr.classes_,'Probability Percentage %':emotion_prob}
    df = pd.DataFrame(data=d,index=None)
   
    
    
    sns.barplot(x=emotion_prob,y=pipe_lr.classes_)
    st.pyplot(fig)

    st.dataframe(df)

   
    

    if option=='Yes':
        summarizer = pipeline("summarization")


        st.header("Summary:")
        summary = summarizer(input, max_length=300, min_length=50, do_sample=False)
        st.write(summary[0]["summary_text"])
        

    








