import streamlit as st
import langchain
from openai import OpenAI
from langchain.llms import OpenAI
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch


st.title("🛫 Airline experience")

prompt = st.text_input("Share with us your experience of the latest trip. ")

my_secret_key = st.secrets["MyOpenAIKey"]
os.environ["openai_api_key"] = my_secret_key

llm = ChatOpenAI(openai_api_key=my_secret_key, model="gpt-4o-mini")


prompt_syst1 = """You are a travel agent, who is a specializes on customer experience, and you are going to analyze the experience from the users and provide 3 types of responses.
From the text provided next, you are going to determine:
1) Whether the user had a negative experience and is fault of the airline, for which you will answer "airline_negative" 
2) Whether the user had negative experience and is no fault of the airline (such as arriving late to the airport), for which you will answer "non_airline_negative"
3) Whether the user had a positive experience, for which you will answer "positive".

Text:
{experience_user}

"""

flight_chain = (
        PromptTemplate.from_template(prompt_syst1)
        | llm
        | StrOutputParser()
    )

positive_chain = PromptTemplate.from_template(
    """You are a travel agent, who is a specializes on customer experience.\
You should thank them for their feedback and for choosing to fly with the airline.
The tone should be professional. Your answers will be on first person, remember its not a mail, its an interaction with a customer

Text:
{text}

"""
) | llm


airline_negative_chain = PromptTemplate.from_template(
    """You are a travel agent, who is a specializes on customer experience.\
Offer sympathies and inform the user that customer service will contact them soon to resolve the issue or provide compensation

The tone should be professional.Your answers will be on first person, remember its not a mail, its an interaction with a customer
    
Text:
{text}

"""
) | llm


non_airline_negative_chain = PromptTemplate.from_template(
    """You are a travel agent, who is a specializes on customer experience..\
Offer sympathies and give a concise explanation that the airline is not liable in such situations.

The tone should be professional. Your answers will be on first person, remember its not a mail, its an interaction with a customer

Text:
{text}

"""
) | llm

main_chain = PromptTemplate.from_template(
    """You are a travel agent who specicializes on customer experiences, your answers will be on first person, remember its not a mail, its an interaction with a customer.
    Provide as an answer right now the following text: "I am here to help you, please provide your feedback". 

Text:
{text}

"""
) | llm


branch = RunnableBranch(
    (lambda x: "airline_negative" in x["exp_type"].lower(), airline_negative_chain),
    (lambda x: "non_airline_negative" in x["exp_type"].lower(), non_airline_negative_chain),
    (lambda x: "positive" in x["exp_type"].lower(), positive_chain),
    main_chain
)

### Put all the chains together
full_chain = {"exp_type": flight_chain, "text": lambda x: x["experience_user"]} | branch



response=full_chain.invoke({"experience_user": prompt})


### Display
st.write(
    response.content
)