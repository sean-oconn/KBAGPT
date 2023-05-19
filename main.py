
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import DirectoryLoader
import os 
import openai
import pinecone
from langchain.chains.question_answering import load_qa_chain

#Connect to pinecone
pinecone.init(
    api_key='',
    environment='')
index_name=""

#Function to load documents + store them in pinecone, usually skipped unless new data is being added
def loadDocs():
    loader=DirectoryLoader('path to data', glob= '**/*.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OpenAI_API_KEY'])
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings,index_name=index_name)
    return docsearch

#Embeds question, sends it to pinecone, which sends the relevant docs + prompt to OpenAI for analysis
def getAnswer(question):
    llm=OpenAI(temperature=0)
    chain=load_qa_chain(llm,chain_type='stuff')
    docs=docsearch.similarity_search(question)
    return chain.run(input_documents=docs, question=prompt + question)
    
#API Key stuff, only way I could get it to work and I have no clue why
os.environ['OpenAI_API_KEY'] = ''
openai.api_key = ''
prompt='Please answer the following question using only the given information, write any steps in a numbered list, if you dont know the answer, tell them to email canvas@ucsd.edu, include the SOURCE: '


needDocs=input("Do you want to load docs Y/N ")
if needDocs=="Y":
    docsearch=loadDocs()
else:
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OpenAI_API_KEY'])
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

question=None
while question != "Q":
    question=input("How can I help you today? \n")
    if question=='Q':
        break
    print(getAnswer(question))
    print ("\n")
    
