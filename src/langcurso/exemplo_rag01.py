from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # adiciona mais palavras no chunch relativo ao anterior
from langchain_openai import OpenAIEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os 
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent

documents = PyPDFLoader("exemplo_rh.pdf").load()
print(f"Número de páginas: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

docs = text_splitter.split_documents(documents)
print (f"Total de pedaços: {len(docs)}")

persist_directory = "./chroma_db"

embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3.5-embed-8b",
    truncate="END",
)

if os.path.exists(persist_directory):
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("Banco de dados vetorial carregado do disco.")
else:
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    print("Banco de dados vetorial criado e salvo em disco.")

retriever = vector_store.as_retriever(search_kwargs={"k": 2})
llm = ChatNVIDIA(model="nvidia/nemotron-3.5-super-vl-120b-a12b", temperature=0 )

pergunta = input ('Faça uma pergunta sobre o documento do RH')
simular = retriever.invoke (pergunta)
contexto = "\n\n".join([doc.page_content for doc in simular])
#for doc in simular: 
#    contexto += doc.page_content
#print(contexto)
resposta = llm.invoke(
    f"Contexto: \n {contexto}\n\n Pergunta:{pergunta}"
)
print(resposta.content)