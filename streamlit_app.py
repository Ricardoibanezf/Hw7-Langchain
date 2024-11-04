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


prompt_syst1 = """
You are a travel agent specializing in customer experience. Based on the provided text, determine:
1) "airline_negative" if the user had a negative experience due to the airline's fault.
2) "non_fault_airline_negative" if the user had a negative experience and is not the airline's fault, this coudl be because he arrived late or something else.
3) "positive" if the user had a positive experience.

Text:
{experience_user}
"""

flight_chain = (
        PromptTemplate.from_template(prompt_syst1)
        | llm
        | StrOutputParser()
    )

non_airline_negative_chain = PromptTemplate.from_template(
    """You are a travel agent, who is a specializes on customer experience..\
Explain that the airline is not liable in such situations, Nothing else. Dont mention at all that  airline will give provide compensation or anything. 

The tone should be professional. Your answers will be on first person, remember its not a mail, its an interaction with a customer

Text:
{text}

"""
) | llm


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




main_chain = PromptTemplate.from_template(
    """You are a travel agent who specicializes on customer experiences, your answers will be on first person, remember its not a mail, its an interaction with a customer.
    Provide only the following text: "I am here to help you". 

Text:
{text}

"""
) | llm


branch = RunnableBranch(
    (lambda x: x["exp_type"].strip().lower() == "airline_negative", airline_negative_chain),
    (lambda x: x["exp_type"].strip().lower() == "non_fault_airline_negative", non_airline_negative_chain),
    (lambda x: x["exp_type"].strip().lower() == "positive", positive_chain),
    main_chain
)



full_chain = {"exp_type": flight_chain, "text": lambda x: x["experience_user"]} | branch

import langchain
langchain.debug = True

response = full_chain.invoke({"experience_user": prompt})



st.write(response.content)
