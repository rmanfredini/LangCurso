from langchain.chat_models import init_chat_model
from langchain.agents import create_agent 
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_tavily import TavilySearch   
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()

#memória para o contexto
contexto=InMemorySaver()

model = ChatNVIDIA(model="nvidia/nemotron-3.5-super-vl-120b-a12b")
agente_ads = create_agent(model=model,
        system_prompt="Você é um assistente que responde perguntas em português brasileiro, responder somente se tiver certeza de ter a informação",
        tools=[TavilySearch()],
        checkpointer=contexto, 
        
config = {"configurable":{"thread_id":"1"}}
print ("agente em funcionamento")
while True:
    pergunta=input("Digite sua pergunta: ")
    if pergunta=="sair":
        break
    resposta = agente_ads.invoke  ({"messages":[{"role":"user","content": pergunta }]},config=config)
    print(f"Agente: {resposta["messages"][-1].content}")



