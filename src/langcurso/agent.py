from langchain.chat_models import init_chat_model
from langchain.agents import create_agent 
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_tavily import TavilySearch   
load_dotenv()

model = ChatNVIDIA(model="nvidia/nemotron-3.5-super-vl-120b-a12b")
agente_ads = create_agent(model=model,
        system_prompt="Você é um assistente que responde perguntas em português brasileiro, responder somente se tiver certeza de ter a informação",
        tools=[TavilySearch()] 
        )
#pergunta=("Qual a temperatura agora em farroupilha?")
#resposta = agente_ads.invoke  ({"messages":[{"role":"user","content": pergunta }]})
#print(resposta["messages"][-1].text)
