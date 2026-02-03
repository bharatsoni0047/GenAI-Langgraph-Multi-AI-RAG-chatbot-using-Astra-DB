import os
import json
from typing import List, Literal
from typing_extensions import TypedDict
import cassio
from pydantic import BaseModel, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.cassandra import Cassandra
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

from langgraph.graph import StateGraph, START, END

# --------------------------------------------------
# ENV VARIABLES (Docker / Local)
# --------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --------------------------------------------------
# ASTRA DB INIT
# --------------------------------------------------

cassio.init(
    token=ASTRA_DB_TOKEN,
    database_id=ASTRA_DB_ID
)

# --------------------------------------------------
# VECTOR STORE (ASSUMES DATA ALREADY INDEXED)
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None
)

retriever = astra_vector_store.as_retriever()

# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)

# --------------------------------------------------
# WIKIPEDIA TOOL
# --------------------------------------------------

wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=200
)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# --------------------------------------------------
# ROUTER SCHEMA
# --------------------------------------------------

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Route query to vectorstore or wikipedia"
    )

# --------------------------------------------------
# GRAPH STATE
# --------------------------------------------------

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]

# --------------------------------------------------
# GRAPH NODES
# --------------------------------------------------

def retrieve(state: GraphState):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def wiki_search(state: GraphState):
    question = state["question"]
    docs = wiki.invoke({"query": question})
    wiki_doc = Document(page_content=docs)
    return {"documents": [wiki_doc], "question": question}


def generate(state: GraphState):
    question = state["question"]
    docs = state["documents"]

    context = "\n\n".join(
        d[0].page_content if isinstance(d, tuple) else d.page_content
        for d in docs
    )

    prompt = f"""
You are an expert assistant.
Answer the question ONLY using the context.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    return {"generation": response.content}


def route_question(state: GraphState):
    question = state["question"]

    prompt = f"""
You are a router.
Return ONLY valid JSON.

Allowed values:
- vectorstore
- wiki_search

Question: {question}

Output:
{{"datasource": "<value>"}}
"""

    response = llm.invoke(prompt)
    datasource = json.loads(response.content)["datasource"]

    return "wiki_search" if datasource == "wiki_search" else "vectorstore"

# --------------------------------------------------
# LANGGRAPH
# --------------------------------------------------

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("wiki_search", wiki_search)
workflow.add_node("generate", generate)

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("wiki_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# --------------------------------------------------
# PUBLIC FUNCTION (API / UI READY)
# --------------------------------------------------

def run_query(question: str) -> str:
    inputs = {"question": question}
    final_state = None

    for output in app.stream(inputs):
        for _, value in output.items():
            final_state = value

    return final_state["generation"]


# --------------------------------------------------
# LOCAL TEST
# --------------------------------------------------

if __name__ == "__main__":
    print(run_query("What is an AI agent?"))
