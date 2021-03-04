from flask import Flask,url_for,render_template,request
import spacy
from spacy import displacy
from mediawiki import MediaWiki
nlp = spacy.load('en_core_web_md')
import json
import wikipedia
import streamlit as st 
import wikipediaapi
import spacy_streamlit
wiki_wiki = wikipediaapi.Wikipedia('en')

wikipedia1 = MediaWiki()
from spacy_streamlit import visualize_ner







# def analyze_text(text):
# 	return nlp(text)

def index():
    st.title("Intern Project")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Intern Project ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Wiki = st.text_input("wiki","Type Here")
    if st.button("Predict"):
        extract(Wiki)
        
	



def extract(wiki):
       raw_test2 = wikipedia1.page(wiki)
       docx = nlp(raw_test2.summary)
       visualize_ner(docx, labels=nlp.get_pipe("ner").labels)
       
      
      
			


		




if __name__ == '__main__':
	index()