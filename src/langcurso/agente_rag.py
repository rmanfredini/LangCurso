from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # adiciona mais palavras no chunch relativo ao anterior
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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

docs = text_splitter.split_documents(documents)

persist_directory = "./chroma_db"
embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3.5-embed-8b",
    truncate="END",
)

if os.path.exists(persist_directory):
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)

@tool (response_format="content_and_artifact")
def buscar_no_documento(pergunta:str):
    """Essa Ferramenta Busca informações no Documento carregado
    Use sempre que o usuario fizer perguntas relacionadas ao manual do rh da empresa"""
    retrieved_docs = vector_store.similarity_search(pergunta, k=2)
    serialized = "\n\n".join(
        f"Fonte: {doc.metadata}\n Conteudo: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


llm = ChatNVIDIA(model="nvidia/nemotron-3.5-super-vl-120b-a12b", temperature=1 )


agente_rag = create_agent(
    model = llm,
    tools = [buscar_no_documento],
    system_prompt = "Você é um assistente que responde perguntas sobre o que você sabe. Se não souber, diga educadamente que não sabe a resposta. Se a informação estiver no documento, utilize a ferramenta buscar_no_documento para obter a informação."
)
