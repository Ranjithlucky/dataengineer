
import pickle
import time
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.vectorstores import PGVector
import psycopg2

from dotenv import load_dotenv
import os
import streamlit as st
load_dotenv()
from pydantic import BaseModel
from typing import List
import re  # added for tokenization


# PostgreSQL connection parameters
DB_HOST = "openai-pgdb.postgres.database.azure.com"
DB_PORT = "5432"
DB_NAME = "vectordb"
DB_USER = "pgadmin"
DB_PASSWORD = "admin@123"

# Establish connection to PostgreSQL
def create_connection():
    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print("Database connection successful.")
        return connection
    except Exception as e:
        print("Error connecting to database:", e)
        return None

connection = create_connection()
if connection is not None:
    cursor = connection.cursor()
    # Proceed with your database operations
else:
    print("Failed to create database connection.")

 

def clean_and_tokenize(text):
    # Step 1: Remove dates and times
    text = remove_dates_times(text)
    # Step 2: Strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 3: Replace multiple spaces with a single space
    text = re.sub(r'http\s+', ' ', text)
    text =text.replace('$', '')
    
    # Step 4: Split the text into tokens
    tokens = text.split()
    token=' '.join(tokens)
    # print("New-Record:",token)  # Debugging line
    return token

def remove_dates_times(text):
    # Remove dates in formats like YYYY-MM-DD, DD-MM-YYYY, MM/DD/YYYY, etc.
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', text)  # YYYY-MM-DD
    text = re.sub(r'\b\d{2}-\d{2}-\d{4}\b', '', text)  # DD-MM-YYYY
    text = re.sub(r'\b\d{2}/\d{2}/\d{4}\b', '', text)  # MM/DD/YYYY
    text = re.sub(r'\b\d{4}/\d{2}/\d{2}\b', '', text)  # YYYY/MM/DD

    # Remove times in formats like HH:MM:SS, HH:MM, etc.
    text = re.sub(r'\b\d{2}:\d{2}:\d{2}\b', '', text)  # HH:MM:SS
    text = re.sub(r'\b\d{2}:\d{2}\b', '', text)        # HH:MM

    # Remove standalone years (e.g., 2023)
    text = re.sub(r'\b\d{4}\b', '', text)
    return text

def conv2dictionary(page_content):

    res = []
    for sub in page_content.rstrip().split("\n"):
        if ':' in sub:
            res.append(map(str.strip, sub.split(':', 1)))
    res = dict(res)
    return res

def get_val(dictionary, key):
    
    value = dictionary.get(key)
    if value is not None and value != "":
        #print(value)
        return value
    else:
        return None

matchArr = ["contract","income contract","Contracts" ,"Monitoring", "Accounting" ,"invalid accounting","Lease ID","Lease","Payment","PLI"]

def find_strings_in_array(dictionary,key,string_array):
    
    input_string = dictionary.get(key)
    if input_string is not None and input_string != " ":    
        found_strings = []
        for element in string_array:
            if element in input_string:
                found_strings.append(element)
                #print(found_strings)
        return found_strings



DB_PATH ="vectorstore\db"
llm = AzureChatOpenAI(    
            deployment_name=os.getenv('OPENAI_DEPLOYMENT_NAME'),    
            #openai_api_type="azure",        
            streaming=True,
            #max_tokens=3000,
            temperature=os.getenv('TEMPERATURE'),
    
 )

embeddings =  OpenAIEmbeddings(deployment=os.getenv('AZURE_OPENAI_ADA_DEPLOYMENT'),)

def clear_history():

    if 'history' in st.session_state:
        del st.session_state['history']




st.title('Chat with CSV Document')

uploaded_file = st.file_uploader('Upload a file:',type=['csv','pdf'])

add_file = st.button('Add File', on_click=clear_history)


if uploaded_file is not None and add_file:
    with st.spinner('Reading, chunking and embedding file...'):
        bytes_data = uploaded_file.read()
        file_name = os.path.join('.', uploaded_file.name)
        with open(file_name,'wb') as f:
            f.write(bytes_data)

        name, extension = os.path.splitext(file_name)

        # Connect to PostgreSQL
        connection = create_connection()
        cursor = connection.cursor()

        if extension =='.csv':
            # ticketres = {'content':'Number'}
            
       
            loader = CSVLoader(file_name,encoding="ISO-8859-1")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            #print(len(chunks))
            
            db =None
            vectorstore = None
            batch_size =5 # Define your preferred batch size
            meta_chunk_batch=[]

    

            for i in range(0, len(chunks),batch_size):
                chunk_batch = chunks[i:(i + batch_size)]
       
                for chunk in chunk_batch:
                    # pg_cnt =chunk.page_content
                    # pg_cnt = clean_and_tokenize(pg_cnt)
                    cleaned_text=clean_and_tokenize(chunk.page_content)
                    embedding = embeddings.embed_documents([cleaned_text])[0]  # Generate embedding
 
                # Insert into PostgreSQL
                    cursor.execute("INSERT INTO items (content, embedding) VALUES (%s, %s)"(cleaned_text, embedding))
                    
        elif extension =='.pdf':
            vectorstore = None
            #loader = UnstructuredPDFLoader(file_name, mode='single', strategy='fast',)
            loader = PyMuPDFLoader(file_name)
            print("Loading raw document..." + loader.file_path)
            raw_documents = loader.load()

            print("Splitting text...")
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=800,
                chunk_overlap=100,
                length_function=len,
            )
            documents = text_splitter.split_documents(raw_documents)

            for document in documents:
                cleaned_text = clean_and_tokenize(document.page_content)
                embedding = embeddings.embed_documents([cleaned_text])[0]  # Generate embedding
 
                # Insert into PostgreSQL
                cursor.execute(
                    "INSERT INTO items (content, embedding) VALUES (%s, %s)",
                    (cleaned_text, embedding)
                )
 
        connection.commit()
        cursor.close()
        connection.close()
        print("Documents saved to database with embeddings.")
 
            

            
