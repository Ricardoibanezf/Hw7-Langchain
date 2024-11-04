import streamlit as st
from openai import OpenAI
import os

st.title("🎈 Airline experience")

prompt = st.text_input("Share with us your experience of the latest trip. ")

my_secret_key = st.secrets["MyOpenAIKey"]
os.environ["openai_api_key"] = my_secret_key

llm = ChatOpenAI(openai_api_key=my_secret_key, model="gpt-4o-mini")