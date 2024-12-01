import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama 
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true" 
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

## Prompt template

system_message = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant. Please respond to the question asked."
)
user_message = HumanMessagePromptTemplate.from_template(
    "Question: {question}"
)
prompt = ChatPromptTemplate.from_messages([system_message, user_message])

## Streamlit Framework

st.title("Simple LangChain Chatbot with Gemma:2b from Ollama")
# st.sidebar.header("LangChain Chatbot")

input_text = st.text_input("Ask a question")


## Ollama gemma:2b model

llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke( {"question": input_text} ))
    


## We weould get warning for Ollama version used so use 

#  from langchain_ollama import OllamaLLM and llm = OllamaLLM(model="gemma:2b") because the class `Ollama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0.

    