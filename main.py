import os
import logging
from typing import TypedDict, List

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

# Configureer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Laad de API-sleutel vanuit het .env-bestand
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logging.error("OPENAI_API_KEY is niet gevonden in het .env-bestand.")
    raise ValueError("OPENAI_API_KEY ontbreekt!")

# Definieer de structuur van de state
class State(TypedDict):
    messages: List[str]

# Node: Verwerk de gebruikersinput en genereer een standaardantwoord
def verwerk_input(state: State) -> dict:
    """
    Verwerkt de gebruikersinput en voegt een standaardantwoord toe.
    """
    user_message = state["messages"][-1]
    antwoord = f"Je zei: '{user_message}'. Hoe kan ik je verder helpen?"
    nieuwe_berichten = state["messages"] + [antwoord]
    logging.info("Antwoord gegenereerd: %s", antwoord)
    return {"messages": nieuwe_berichten}

# (Optioneel) Node: Genereer een dynamische reactie via een LLM
def genereer_reactie(state: State) -> dict:
    """
    Roept een LLM aan om een dynamische reactie te genereren op basis van de context.
    """
    llm = ChatOpenAI(api_key=openai_api_key, model="openai/gpt-4-turbo-preview", temperature=0.7)
    context = "\n".join(state["messages"])
    reactie = llm.invoke(context)
    nieuwe_berichten = state["messages"] + [reactie]
    logging.info("LLM reactie gegenereerd.")
    return {"messages": nieuwe_berichten}

# Bouw de LangGraph
def build_graph() -> StateGraph:
    """
    Bouwt de LangGraph-workflow en retourneert de gecompileerde applicatie.
    """
    graph_builder = StateGraph(State)
    # Voeg hier nodes toe; in dit voorbeeld gebruiken we de standaard 'verwerk_input'
    graph_builder.add_node("verwerk_input", verwerk_input)
    # Indien gewenst, kun je ook de LLM-node toevoegen:
    # graph_builder.add_node("genereer_reactie", genereer_reactie)

    # Stel de start- en eindpunten in
    graph_builder.set_entry_point("verwerk_input")
    graph_builder.set_finish_point("verwerk_input")
    
    return graph_builder.compile()

# Initialiseer de FastAPI app
app = FastAPI(title="Lavans BV Bedrijfsassistent")

# Mount een map voor statische bestanden (optioneel)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configureer Jinja2Templates (maak een map 'templates' in je project)
templates = Jinja2Templates(directory="templates")

# Maak de LangGraph-applicatie aan
graph_app = build_graph()

@app.get("/", response_class=HTMLResponse)
async def lees_index(request: Request):
    """
    Render de hoofdpagina met een eenvoudig invoerformulier.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/verwerk", response_class=JSONResponse)
async def verwerk_bericht(bericht: str = Form(...)):
    """
    Verwerkt het ingevoerde bericht via de LangGraph-workflow en retourneert het antwoord als JSON.
    """
    # Initialiseer de state met het gebruikersbericht
    initial_state = {"messages": [bericht]}
    final_state = graph_app.invoke(initial_state)
    # Retourneer de volledige conversatie als JSON
    return JSONResponse(content={"messages": final_state["messages"]})

# Voor een eenvoudige debug-run kun je de app met uvicorn starten:
# uvicorn main:app --reload
