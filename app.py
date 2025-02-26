from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv 

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

#Langsmith tracking
os.environ["LANGCHAIN_TRACKING"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


#prompt template

prompt = ChatPromptTemplate.from_messages(

    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)


#streamlit Framework
st.title('Langchain Demo with OPENAI API')
input_text = st.text_input("Search the topic u want.")


llm = ChatOpenAI(model="gpt-4o")
output_parser = StrOutputParser()

chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))