#from langchain.llms import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import magic
import pickle
import os 
import nltk
import string
import openai
import re

openai.api_key = ''
model_engine = "text-davinci-002"
#has_data controls weather or not the program will chunk and create vectors for a text document or use the last save of the vectors. If no new data is being uploaded in between runs, keep it set to TRUE so save time!
has_data=True
# Define a function to clean up the answer text
def clean_answer(answer):
#Answers haven't been displaying any of this anyway, keeping it in just in case it starts when more data is loaded
    # Remove any leading or trailing white space
    #answer = answer.strip()
    
    # Remove any extra line breaks or tabs
    #answer = re.sub(r'\n+', '\n', answer)
    #answer = re.sub(r'\t+', '\t', answer)
    
    # Remove any URLs or email addresses
    #answer = re.sub(r'http\S+', '', answer)
    #answer = re.sub(r'\S+@\S+', '', answer)
    
    # Remove any punctuation except for periods, question marks, and exclamation points
    #answer = answer.translate(str.maketrans('', '', string.punctuation.replace('.?!', '')))
    
    # Use ChatGPT to normalize the answer text
    normalized_answer = openai.Completion.create(
        engine=model_engine,
        prompt= "make this sound warm and friendly, make this responce suitable for an end user to follow a numbered instructions: " + answer ,
        max_tokens=1024,
        temperature=.1,
        n = 1,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0).choices[0].text.strip()
    
    return normalized_answer

os.environ['OPENAI_API_KEY']=''

if has_data==False:
    # Load documents from a directory
    loader=DirectoryLoader('path to documents', glob= '**/*.txt')
    documents=loader.load()

    # Split the documents into smaller text chunks
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts=text_splitter.split_documents(documents)
    # Embed the text using OpenAI
    embeddings=OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    # Convert the embedded text into a searchable index
    docsearch=Chroma.from_documents(texts,embeddings)

else:
    with open("embeddings.pkl", "rb") as f:
       embeddings = pickle.load(f)
    with open("texts.pkl", "rb") as f:
       texts = pickle.load(f)
    docsearch=Chroma.from_documents(texts,embeddings)
 



# Create an instance of the RetrievalQA class using OpenAI as the language model
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())
# Define a query and run the question answering system
query="how can I adjust assignment group weights "

answer=qa.run(query)
print ("OG" + answer + '\n')
# Clean up the answer text and print it
print(clean_answer(answer))

if has_data==False:
    with open("embeddings.pkl", "wb") as f:
       pickle.dump(embeddings,f)
    with open("texts.pkl" , "wb") as f:
       pickle.dump(texts,f)


