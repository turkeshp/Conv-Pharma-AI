Final _2
# -*- coding: utf-8 -*-
"""
Supply Chain Expert Chatbot with Dynamic SPARQL Query Generation, Multilingual Support,
Enhanced Explainability, Improved NER, Force-Directed Graph Visualization,
Auto-Scrolling Chat, and Reset Chat Functionality
"""
 
import os
import gradio as gr
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import networkx as nx
import plotly.graph_objects as go  # For graph visualization
from langchain_openai import ChatOpenAI  # Updated import
from langchain.memory import ConversationBufferMemory
import requests
import logging
import random
import spacy
from transformers import pipeline  # For transformers
from deep_translator import GoogleTranslator
from typing import List, Tuple
 
# === Configure Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
 
# === Load spaCy Model ===
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded successfully.")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model downloaded and loaded successfully.")
 
# === Initialize BERT NER Pipeline ===
ner_pipeline = None
try:
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    logger.info("BERT NER model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading BERT NER model: {e}")
 
# === Set Environment Variables ===
FUSEKI_ENDPOINT = 'http://localhost:3030/thesis_1'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")
 
# === Initialize OpenAI via LangChain's ChatOpenAI ===
langchain_openai = ChatOpenAI(
    temperature=0,
    model="ft:gpt-4o-2024-08-06:personal:supply-chain-expert:A22AlOWt",  # 
    openai_api_key=OPENAI_API_KEY
)
 
# === SentenceTransformer Model for FAISS ===
model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("SentenceTransformer model loaded successfully.")
 
# === Load the FAISS Index and Data ===
faiss_index_path = 'faiss_index.bin'
if not os.path.exists(faiss_index_path):
    raise FileNotFoundError(f"FAISS index file '{faiss_index_path}' not found.")
index = faiss.read_index(faiss_index_path)
logger.info("FAISS index loaded successfully.")
 
# === Load the Question and Answer Data ===
required_files = ['questions.pkl', 'answers.pkl']
for file_name in required_files:
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Required file '{file_name}' not found.")
 
with open('questions.pkl', 'rb') as f:
    questions = pickle.load(f)
logger.info("Questions loaded successfully.")
 
with open('answers.pkl', 'rb') as f:
    answers = pickle.load(f)
logger.info("Answers loaded successfully.")
 
# === LangChain Memory Initialization ===
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
logger.info("Conversation memory initialized.")
 
# === Handle Casual Greetings ===
def handle_greeting(query):
    greetings = ["hi", "hello", "hey", "greetings", "howdy"]
    if query.lower() in greetings:
        return "Hello! How can I assist you with your supply chain inquiries today?"
    return None
 
# === Function to Preprocess Query Based on User-Selected Language ===
def preprocess_query(query: str, selected_lang: str) -> Tuple[str, str]:
    """
    Translates the query to English if the selected language is not English.
    Returns the language code and the (possibly translated) query.
    """
    try:
        if selected_lang.lower() != 'english' and selected_lang != 'en':
            # Map language names to language codes if necessary
            lang_code = {
                'Spanish': 'es',
                'French': 'fr',
                'German': 'de',
                'Italian': 'it'
            }.get(selected_lang, 'en')
            translated = GoogleTranslator(source='auto', target='en').translate(query)
            logger.info(f"Translated query: {translated}")
            return lang_code, translated
        else:
            return 'en', query
    except Exception as e:
        logger.error(f"Error in language detection/translation: {e}")
        # Assume English if translation fails
        return 'en', query
 
# === Function to Classify Query Intent Using OpenAI ===
def classify_query_intent(query):
    """
    Uses OpenAI's GPT model to classify the user's query intent.
    Returns the intent as 'sparql', 'faiss', or 'hybrid'.
    """
    # Check for greetings
    greeting_response = handle_greeting(query)
    if greeting_response:
        return 'greeting'
 
    # === Prompt Engineering Addition ===
    # Refined prompt with Chain-of-Thought to enhance classification accuracy
    refined_prompt = f"""
You are a helpful assistant that classifies user queries into one of the following categories: 'sparql', 'faiss', or 'hybrid'.
 
Here are examples of each category:
 
- sparql: "Retrieve the supply chain details for Pfizer in Europe."
- faiss: "What are the negotiation strategies with Sanofi?"
- hybrid: "Explain the collaboration between Pfizer and Moderna."
 
Classify the following user query into one of these categories and output only the category name in lowercase.
Provide a brief explanation of your reasoning process.
 
Query: "{query}"
 
Category:
"""
 
    response = langchain_openai.invoke(refined_prompt)
    response_text = response.content.strip().lower()
    logger.info(f"Intent classified as: {response_text}")
 
    if response_text in ['sparql', 'faiss', 'hybrid']:
        return response_text
    else:
        return 'hybrid'  # Default to hybrid if the classification is unclear
 
# === Function to Extract Entities and Relations for SPARQL Query Generation ===
def extract_entities_relations(query: str) -> Tuple[List[str], List[str]]:
    """
    Extracts entities and possible relations from the query using spaCy and transformers.
    Returns lists of entities and relations.
    """
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT", "PERSON", "FAC"]]
 
    # Use BERT-based NER model for additional entities
    if ner_pipeline is not None:
        ner_results = ner_pipeline(query)
        bert_entities = [result['word'] for result in ner_results if result['entity'] in ['B-ORG', 'B-LOC', 'B-PER']]
        entities.extend(bert_entities)
    else:
        logger.warning("BERT NER pipeline is not available. Skipping BERT-based entity extraction.")
 
    # Remove duplicates
    entities = list(set(entities))
 
    logger.info(f"Extracted entities: {entities}")
 
    # Assume nouns as relations, excluding entities
    relations = [token.text for token in doc if token.pos_ == "NOUN" and token.text not in entities]
    logger.info(f"Extracted relations: {relations}")
 
    # === Prompt Engineering Addition ===
    # Supplementary prompt to enhance entity extraction using Chain-of-Thought
    supplementary_prompt = f"""
You are an assistant that extracts entities and their relationships from user queries.
 
User Query: "{query}"
 
List any additional entities (e.g., organizations, locations, products) mentioned in the query:
 
Entities:
"""
 
    supplementary_response = langchain_openai.invoke(supplementary_prompt)
    additional_entities = supplementary_response.content.strip().split('\n')
    additional_entities = [ent.strip() for ent in additional_entities if ent.strip()]
    entities.extend(additional_entities)
    entities = list(set(entities))  # Remove any new duplicates
 
    logger.info(f"Additional entities extracted: {additional_entities}")
    logger.info(f"Updated entities: {entities}")
 
    return entities, relations
 
# === Function to Generate Dynamic SPARQL Query ===
def generate_sparql_query(query: str) -> str:
    """
    Generates a SPARQL query based on extracted entities and relations.
    """
    entities, relations = extract_entities_relations(query)
 
    # Construct WHERE clauses based on entities and relations
    where_clauses = []
    for entity in entities:
        entity_uri = f"ex:{entity.replace(' ', '_')}"
        for relation in relations:
            relation_uri = f"ex:{relation.replace(' ', '_')}"
            where_clause = f"{entity_uri} {relation_uri} ?value."
            where_clauses.append(where_clause)
 
    if not where_clauses:
        # Default query if no entities or relations found
        sparql_query = """
PREFIX ex: <http://example.org/pharma-supply-chain#>
SELECT ?subject ?predicate ?object WHERE {
    ?subject ?predicate ?object.
} LIMIT 10
"""
    else:
        where_statements = "\n".join(where_clauses)
        sparql_query = f"""
PREFIX ex: <http://example.org/pharma-supply-chain#>
SELECT ?value WHERE {{
{where_statements}
}}
"""
    logger.info(f"Generated SPARQL query: {sparql_query}")
    return sparql_query
 
# === Function to Execute SPARQL Query in Fuseki ===
def execute_sparql_query(query):
    try:
        headers = {"Content-Type": "application/sparql-query", "Accept": "application/sparql-results+json"}
        response = requests.post(
            f"{FUSEKI_ENDPOINT}/query",
            headers=headers,
            data=query
        )
        if response.status_code == 200:
            results = response.json()
            formatted_results = []
            for result in results['results']['bindings']:
                values = [value['value'] for value in result.values()]
                formatted_results.append(" - ".join(values))
            logger.info("SPARQL Query executed successfully.")
            return formatted_results
        else:
            logger.error(f"SPARQL query failed with status code {response.status_code}: {response.text}")
            return [f"SPARQL query failed with status code {response.status_code}"]
    except Exception as e:
        logger.error(f"Error during SPARQL execution: {e}")
        return [f"Error during SPARQL execution: {e}"]
 
# === FAISS Chain with Explainability ===
def execute_faiss_chain_with_explanation(query):
    logger.debug(f"Encoding query: {query}")
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding).astype(np.float32), k=5)
    logger.debug(f"FAISS search distances: {D}")
    logger.debug(f"FAISS search indices: {I}")
 
    results = []
    explanations = []
    for i, idx in enumerate(I[0]):
        if idx < len(answers):
            document_id = f"DOC-{random.randint(1000, 9999)}"  # Randomly generated document ID
            relevance_score = float(D[0][i])
            result = {
                'id': idx,
                'document_id': document_id,
                'relevance_score': relevance_score,
                'result': answers[idx]
            }
            results.append(result)
            explanations.append(f"Document: {document_id} | Relevance Score: {relevance_score:.10f}")
    logger.info(f"FAISS Search Results: {results}")
    return results, explanations
 
# === Function to Generate a Response Using LangChain ===
def generate_response(query, retrieved_data):
    # Create a prompt
    prompt = f"""
You are a supply chain expert assistant.
 
User Query: "{query}"
 
Retrieved Data:
{retrieved_data}
 
Please provide a helpful and concise answer to the user's query based on the retrieved data.
"""
 
    # === Prompt Engineering Addition ===
    # Enhanced prompt with Chain-of-Thought for better reasoning and explainability
    refined_prompt = f"""
You are a supply chain expert assistant.
 
User Query: "{query}"
 
Retrieved Data:
{retrieved_data}
 
Please provide a helpful and concise answer to the user's query based on the retrieved data.
Explain your reasoning process step-by-step to enhance transparency and reliability.
 
Response:
"""
 
    response = langchain_openai.invoke(refined_prompt)
    response_text = response.content.strip()
    return response_text
 
# === Function to Generate a Force-Directed Graph ===
def generate_force_directed_graph(entities, relations):
    G = nx.Graph()
 
    # Define entity types for coloring (assuming entity types are known; modify as needed)
    entity_types = {}
    for ent in entities:
        doc = nlp(ent)
        for token in doc:
            if token.ent_type_:
                entity_types[ent] = token.ent_type_
                break
        else:
            entity_types[ent] = 'OTHER'
 
    # Color mapping for entity types
    color_map = {
        'ORG': '#1f77b4',
        'GPE': '#ff7f0e',
        'PRODUCT': '#2ca02c',
        'PERSON': '#d62728',
        'FAC': '#9467bd',
        'OTHER': '#8c564b'
    }
 
    # Add nodes with color based on entity type
    for entity in entities:
        G.add_node(entity, color=color_map.get(entity_types.get(entity, 'OTHER'), '#8c564b'))
 
    # Add edges
    for relation in relations:
        # For simplicity, connect all entities with the relation
        for entity in entities:
            target = f"{relation}"
            G.add_node(target, color='#17becf')  # Relations have a default color
            G.add_edge(entity, target)
 
    # Generate layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)
 
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
 
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
 
    # Create node traces
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(G.nodes[node]['color'])
        node_text.append(f"{node} ({G.nodes[node].get('color', 'N/A')})")
 
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(
            showscale=False,  # Disable color scale bar
            color=node_colors,
            size=[15 if node in entities else 10 for node in G.nodes()],
            line_width=2
        )
    )
 
    # Define the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Knowledge Graph Visualization',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper"
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
 
    # Enhance interactivity
    fig.update_layout(
        clickmode='event+select',
        dragmode='pan'
    )
 
    return fig
 
# === Function to Generate and Display Graph Based on User Query ===
def generate_graph_based_on_query(history):
    if not history:
        # Create a Plotly figure with a message
        fig = go.Figure()
        fig.update_layout(
            title="No previous query to generate graph from.",
            xaxis={'visible': False},
            yaxis={'visible': False},
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)',
        )
        fig.add_annotation(
            text="Please enter a query to generate the knowledge graph.",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
 
    last_query = history[-1][0]  # Assuming last user input is at history[-1][0]
    entities, relations = extract_entities_relations(last_query)
   
    if entities and relations:
        fig = generate_force_directed_graph(entities, relations)
        return fig
    else:
        # Create a Plotly figure with a message
        fig = go.Figure()
        fig.update_layout(
            title="No entities or relations found for visualization.",
            xaxis={'visible': False},
            yaxis={'visible': False},
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)',
        )
        fig.add_annotation(
            text="No entities or relations extracted from the query to generate the graph.",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
 
# === Query Execution Chain with Dynamic Classification ===
def handle_query_langchain(query, history, selected_lang):
    # Preprocess query (translation based on selected language)
    lang_code, processed_query = preprocess_query(query, selected_lang)
 
    # Handle greetings
    greeting_response = handle_greeting(processed_query)
    if greeting_response:
        # Translate back if necessary
        if lang_code != 'en':
            translated_response = GoogleTranslator(source='en', target=lang_code).translate(greeting_response)
        else:
            translated_response = greeting_response
        history.append((query, translated_response))
        return translated_response, history
 
    classification = classify_query_intent(processed_query)
    if classification == "sparql":
        sparql_query = generate_sparql_query(processed_query)
        results = execute_sparql_query(sparql_query)
        retrieved_data = "\n".join(results)
        explainability = "SPARQL query executed based on extracted entities and relations."
    elif classification == "faiss":
        results, explanations = execute_faiss_chain_with_explanation(processed_query)
        retrieved_data = "\n".join([res['result'] for res in results])
        explainability = "\n".join(explanations)
    elif classification == "hybrid":
        sparql_query = generate_sparql_query(processed_query)
        sparql_results = execute_sparql_query(sparql_query)
        faiss_results, faiss_explanations = execute_faiss_chain_with_explanation(processed_query)
        retrieved_data = "\n".join(sparql_results + [res['result'] for res in faiss_results])
        explainability = "SPARQL and FAISS queries executed.\n\nFAISS Explainability:\n" + "\n".join(faiss_explanations)
    else:
        retrieved_data = "No relevant data found."
        explainability = "Unable to classify the query intent."
 
    # Generate a response using LangChain and OpenAI
    response_text = generate_response(processed_query, retrieved_data)
 
    # Include explainability in the response
    full_response = f"{response_text}\n\n**Explainability:**\n{explainability}"
 
    # Translate the response back to the user's language if necessary
    if lang_code != 'en':
        translated_response = GoogleTranslator(source='en', target=lang_code).translate(full_response)
    else:
        translated_response = full_response
 
    history.append((query, translated_response))
    return translated_response, history
 
# === Define the Chatbot Response Function for Gradio ===
def chatbot_response(user_input, history, selected_lang):
    try:
        response, history = handle_query_langchain(user_input, history, selected_lang)
        return history, ""  # Update Chatbot and clear textbox
    except Exception as e:
        logger.error(f"An error occurred in chatbot_response: {e}")
        history.append((user_input, "An error occurred. Please try again later."))
        return history, ""
 
# === Function to Reset Chat ===
def reset_chat():
    return [], ""  # Clear history and clear textbox
 
# === Create the Gradio Interface with Built-In Themes ===
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("""
    <div style="text-align: center;">
        <img src="logo.png" style="width:100px;" alt="Pharma Logo"/>
        <h1>Pharma Supply Chain Assistant</h1>
        <p>Your AI assistant for all supply chain queries. Ask anything!</p>
    </div>
    """)
 
    # Language Selection
    language_dropdown = gr.Dropdown(
        choices=["English", "Spanish", "French", "German"],
        label="Select Language",
        value="English",
        info="Select your preferred language."
    )
 
    chatbot = gr.Chatbot(show_label=True, height=400)
 
    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Type your query here...",
            lines=1
        )
        send_button = gr.Button("Send", variant="primary")
        generate_graph_button = gr.Button("Generate Graph", variant="secondary")
        reset_button = gr.Button("Reset Chat", variant="secondary")
 
    state = gr.State([])  # Maintain conversation history
 
    # Event to update chatbot response based on language dropdown
    send_button.click(
        fn=chatbot_response,
        inputs=[txt, state, language_dropdown],
        outputs=[chatbot, txt]
    )
    txt.submit(
        fn=chatbot_response,
        inputs=[txt, state, language_dropdown],
        outputs=[chatbot, txt]
    )
 
    # Generate Graph Button Interaction
    graph_output = gr.Plot(label="Knowledge Graph")
    generate_graph_button.click(
        fn=generate_graph_based_on_query,
        inputs=[state],
        outputs=graph_output
    )
 
    # Reset Chat Button Interaction
    reset_button.click(
        fn=reset_chat,
        inputs=None,
        outputs=[chatbot, txt]
    )
 
    # Note: Gradio's Chatbot component automatically handles auto-scrolling to the latest message.
    # Ensure that the Chatbot receives the complete history list for proper functionality.
 
    gr.Markdown("""
    <style>
        /* Minimal styling using built-in themes; avoid custom CSS */
    </style>
    """)
 
demo.launch(share=True)