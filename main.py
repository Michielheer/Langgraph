from typing import TypedDict, List
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Laad de API-sleutel vanuit het .env-bestand
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is niet gevonden in het .env-bestand.")

# Definieer de structuur van de state
class State(TypedDict):
    messages: List[str]

# Node: Verwerk de gebruikersinput en genereer een antwoord
def verwerk_input(state: State) -> dict:
    # Haal het laatste bericht op van de gebruiker
    user_message = state["messages"][-1]
    # Maak een standaardantwoord
    antwoord = f"Je zei: '{user_message}'. Hoe kan ik je verder helpen?"
    nieuwe_berichten = state["messages"] + [antwoord]
    return {"messages": nieuwe_berichten}

# Optioneel: Node met dynamische reactie via een LLM (kun je later toevoegen)
def genereer_reactie(state: State) -> dict:
    llm = ChatOpenAI(api_key=openai_api_key, model="openai/gpt-4-turbo-preview", temperature=0.7)
    # Combineer de berichten als context
    context = "\n".join(state["messages"])
    reactie = llm.invoke(context)
    nieuwe_berichten = state["messages"] + [reactie]
    return {"messages": nieuwe_berichten}

# Bouw de graf
graph_builder = StateGraph(State)

# Voeg de node toe; in dit voorbeeld gebruiken we alleen 'verwerk_input'
graph_builder.add_node("verwerk_input", verwerk_input)

# Stel de start- en eindpunten in (hier is dat dezelfde node)
graph_builder.set_entry_point("verwerk_input")
graph_builder.set_finish_point("verwerk_input")

# Compileer de graf tot een uitvoerbare applicatie
app = graph_builder.compile()

# Initialiseer de state met een gebruikersbericht
initial_state = {"messages": ["Hallo, ik heb een vraag."]}
final_state = app.invoke(initial_state)

# Toon het resultaat
print("Chatbot reacties:")
for bericht in final_state["messages"]:
    print(bericht)
