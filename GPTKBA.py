
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import os 
import openai


#API Key stuff, only way I could get it to work and I have no clue why
os.environ['OpenAI_API_KEY'] = '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'
openai.api_key = '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'

prompt='Please answer the following question using only the given information, write any steps in a numbered list, if you dont know the answer, tell them to email canvas@ucsd.edu: '
loader=DirectoryLoader('C:/Users/Sean/source/repos/GPTKBA', glob= '**/*.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OpenAI_API_KEY'])
docsearch = Chroma.from_documents(texts, embeddings)


qa=RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())
question=None
while question != "Q":
    question=input("How can I help you today? \n")
    if question=='Q':
        exit
    print(qa.run(prompt + question))
    print ("\n")
    