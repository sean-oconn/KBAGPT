#from langchain.llms import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import magic
import os 
import nltk
import string
import openai
import re

openai.api_key = ''
model_engine = "text-davinci-002"

# Define a function to clean up the answer text
def clean_answer(answer):
    # Remove any leading or trailing white space
    answer = answer.strip()
    
    # Remove any extra line breaks or tabs
    answer = re.sub(r'\n+', '\n', answer)
    answer = re.sub(r'\t+', '\t', answer)
    
    # Remove any URLs or email addresses
    answer = re.sub(r'http\S+', '', answer)
    answer = re.sub(r'\S+@\S+', '', answer)
    
    # Remove any punctuation except for periods, question marks, and exclamation points
    answer = answer.translate(str.maketrans('', '', string.punctuation.replace('.?!', '')))
    
    # Use ChatGPT to normalize the answer text
    normalized_answer = openai.Completion.create(
        engine=model_engine,
        prompt='Add some fluff before the instructions start, and a use a numbered list for the instructions: ' + answer ,
        max_tokens=1024,
        temperature=.3,
        n = 1,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0).choices[0].text.strip()
    
    return normalized_answer

os.environ['OPENAI_API_KEY']=''

# Load documents from a directory
loader=DirectoryLoader('', glob= '**/*.txt')
documents=loader.load()
# Split the documents into smaller text chunks
text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts=text_splitter.split_documents(documents)

# Embed the text using OpenAI
embeddings=OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

# Convert the embedded text into a searchable index
docsearch=Chroma.from_documents(texts,embeddings)

# Create an instance of the RetrievalQA class using OpenAI as the language model
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

# Define a query and run the question answering system
query="I am trying to add a student as an observer to my course but I do have the + People in Canvas.Is this something you can help me with?"

answer=qa.run(query)
# Clean up the answer text and print it
cleaned_answer=clean_answer(answer)
print(cleaned_answer)


