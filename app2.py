from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document

model_local = ChatOllama(model="llama3")




# we can declare extension, display progress bar, use multithreading
loader = DirectoryLoader('/home/lxuser/websites/story', glob="*.html", show_progress=True, use_multithreading=True)

docs = loader.load()
docs = docs[0]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
#doc_splits = text_splitter.split_documents(docs_list)

doc_splits = text_splitter.create_documents(docs_list)

print(len(docs)) # 1 

# 2. Convert documents to Embeddings and store them
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings.ollama.OllamaEmbeddings(model='mxbai-embed-large'), persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever()