import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time as t

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACKING"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
groq_api_key=os.environ["GROQ_API_KEY"]



if "vector" not in st.session_state:
    st.session_state.embeddings=OpenAIEmbeddings()
    st.session_state.loader= WebBaseLoader("https://python.langchain.com/docs/introduction/")
    st.session_state.docs=st.session_state.loader.load()

    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


st.title("ChatGroq")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="qwen-2.5-32b")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Question: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriver = st.session_state.vectors.as_retriever()
reterival_chain = create_retrieval_chain(retriver, document_chain)

prompt = st.text_input("Input your prompt here...")

if prompt:
    start = t.process_time()
    response = reterival_chain.invoke({"input": prompt})
    print("Response time :", t.process_time()-start)
    st.write(response['answer'])


    with st.expander("Document similarity search"):

        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("------------------------")