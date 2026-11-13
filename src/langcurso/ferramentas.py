from langchain.chat_models import init_chat_model
from langchain.agents import create_agent 
from langchain.tools import tool    
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv

load_dotenv() 

   


@tool
def calcular_frete_sedex(peso_kg: float, distancia_km: float) -> float:
    """
    Calcula o valor do frete Sedex

    Use quando o usuário pedir cálculo de frete, envio e entrega
    
    Args:
        peso_kg: peso do produto em quilogramas
        distancia_km: distância em quilômetros
    
    Returns:
        valor do frete em reais
    """
    valor_base = 15.00
    valor_por_kg = 2.50
    valor_por_km = 0.50
    
    return valor_base + (valor_por_kg * peso_kg) + (valor_por_km * distancia_km)

@tool
def converter_temperatura(temperatura: float, unidade: str) -> str:
    """
    Converte temperatura entre Celsius e Fahrenheit

    Use quando o usuário pedir conversão de temperatura
    
    Args:   
        temperatura: temperatura em graus a ser convertida
        unidadde: unidade de temperatura ("celsius" ou "fahrenheit")
    
    Returns:
        temperatura convertida
    """
    if unidade == "celsius":
        return f'Saida --> {(temperatura - 32) * 5 / 9} celsius'
    elif unidade == "fahrenheit":
        return f'Saida -->{(temperatura * 9 / 5) + 32} fahrenheit'
    else:
        raise ValueError("Unidade inválida. Use 'celsius' ou 'fahrenheit'")

model = ChatNVIDIA(model="nvidia/nemotron-3.5-super-vl-120b-a12b")
agente_ads = create_agent(model=model,
        system_prompt="Você é um assistente que responde perguntas em português brasileiro, responder somente se tiver certeza de ter a informação",
        tools=[calcular_frete_sedex,converter_temperatura],
     )

print ("agente em funcionamento")
while True:
    pergunta=input("Digite sua pergunta: ")
    if pergunta=="sair":
        break
    resposta = agente_ads.invoke  ({"messages":[{"role":"user","content": pergunta }]})
    print(f"Agente: {resposta["messages"][-1].content}")
