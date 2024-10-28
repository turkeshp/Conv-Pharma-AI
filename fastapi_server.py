from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import json
import logging

# Import your chatbot functions from app_final_py.py
from app_final_py import handle_query_langchain, generate_graph_based_on_query, reset_chat

# Initialize FastAPI app
app = FastAPI()

# Define Pydantic Models
class QueryRequest(BaseModel):
    user_input: str
    selected_lang: str

class GraphRequest(BaseModel):
    history: List[List[str]]

# FastAPI Endpoints
@app.post("/chat/")
async def chat_endpoint(request: QueryRequest):
    """
    Endpoint to handle user queries and return chatbot responses.
    """
    try:
        history = []  # Initialize history or load a persistent history if needed
        response, history = handle_query_langchain(request.user_input, history, request.selected_lang)
        return {"response": response}
    except Exception as e:
        logging.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Error processing the query")

@app.post("/generate-graph/")
async def generate_graph_endpoint(request: GraphRequest):
    """
    Endpoint to generate and return a knowledge graph based on user query history.
    """
    try:
        fig = generate_graph_based_on_query(request.history)
        return json.loads(fig.to_json())  # Return Plotly figure as JSON
    except Exception as e:
        logging.error(f"Error generating graph: {e}")
        raise HTTPException(status_code=500, detail="Error generating the graph")

@app.post("/reset-chat/")
async def reset_chat_endpoint():
    """
    Endpoint to reset the chat history.
    """
    history = reset_chat()
    return {"history": history}
