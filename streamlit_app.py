import streamlit as st
from openai import OpenAI
from langchain.llms import OpenAI
import os
from langchain_openai import ChatOpenAI


st.title("🛫 Airline experience")

experience_user = st.text_input("Share with us your experience of the latest trip. ")

my_secret_key = st.secrets["MyOpenAIKey"]
os.environ["openai_api_key"] = my_secret_key

llm = ChatOpenAI(openai_api_key=my_secret_key, model="gpt-4o-mini")


prompt_syst1 = """You are a travel agent, who is a specializes on customer experience, and you are going to analyze the experience from the users and provide 3 types of responses.
From the text provided next, you are going to determine whether the user had a negative experience that is fault of the airline, for which you will answer "airline_negative" , a negative experience that is no fault of the airline, for which you will answer "non_airline_negative" or a positive experience, for which you will answer "positive".

Text:
{experience_user}

"""

flight_chain = (
        PromptTemplate.from_template(prompt_syst1)
        | llm
        | StrOutputParser()
    )
