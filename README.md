# Multi-Source RAG Chatbot (LangGraph + AstraDB + Docker)

## Problem Statement
Traditional chatbots often suffer from one of the following limitations:
- They rely only on static internal documents, limiting knowledge scope.
- They depend entirely on external sources, reducing control and reliability.

This project solves the problem by building a **multi-source Retrieval-Augmented Generation (RAG) system** that can intelligently decide:
- when to use **internal domain knowledge** stored in AstraDB (Vector DB)
- when to fall back to **external public knowledge** (Wikipedia)

The entire workflow is explicitly orchestrated using **LangGraph**, making the system production-oriented and controllable.

---

## Project Overview
This is a **production-ready GenAI RAG application**, refactored from an experimental notebook into a **Python service and Dockerized** for reproducible execution.

The system performs the following steps:
1. Accepts a user query
2. Routes the query to the appropriate data source
3. Retrieves relevant context
4. Generates a grounded response using an LLM

---

## High-Level Architecture (LangGraph Flow)

                    ┌──────────────┐
                    │   User Query │
                    └──────┬───────┘
                           │
                           ▼
              ┌────────────────────────┐
              │       Router Node      │
              │     (LLM-based JSON)   │
              └──────────┬─────────────┘
                         │
              ┌──────────┴───────────┐
              ▼                      ▼
      ┌─────────────────┐   ┌─────────────────┐
      │     AstraDB     │   │    Wikipedia    │
      │  Vector Store   │   │   Search Tool   │
      └────────┬────────┘   └────────┬────────┘
               │                     │
               └──────────┬──────────┘
                          ▼
              ┌────────────────────────┐
              │     Generation Node    │
              │      (LLM Answer)      │
              └──────────┬─────────────┘
                         ▼
                ┌──────────────┐
                │ Final Answer │
                 ──────────────┘


## Key Components

### 1. LLM-Based Router (LangGraph)
- Uses an LLM to classify incoming queries
- Routes queries to:
  - AstraDB vector store for domain-specific queries
  - Wikipedia for general knowledge
- Implemented using strict JSON-based routing
- LangGraph conditional edges control execution flow

---

### 2. AstraDB Vector Store (Internal Knowledge)
- Documents are:
  - chunked using recursive text splitting
  - embedded using HuggingFace MiniLM embeddings
- Stored in AstraDB (Cassandra Vector Search)
- Used for controlled, domain-specific retrieval

---

### 3. Wikipedia Tool (External Knowledge)
- Acts as a fallback knowledge source
- Implemented as a separate LangGraph node
- Enables hybrid retrieval across internal and external data sources

---

### 4. Generation Node (RAG Completion)
- Retrieved outputs are normalized into a unified format
- Context is constructed from retrieved documents
- LLM generates a grounded response
- Ensures this is a full RAG pipeline, not a retrieval-only demo

---

### 5. Production Edge-Case Handling
- Handles heterogeneous retrieval outputs:
  - `(Document, score)` tuples from vector databases
  - `Document` objects
  - raw text responses from external tools
- Normalization avoids runtime failures in generation stage

---

## Tech Stack
- Python
- LangGraph (workflow orchestration)
- LangChain (RAG utilities)
- Groq LLM (routing + generation)
- AstraDB (Cassandra Vector DB)
- HuggingFace Embeddings
- Wikipedia API
- Docker

---

## Running the Project (Docker)

### Build Docker Image

- docker build -t multisource-rag .
- End-to-end RAG pipeline implemented
- Multi-source retrieval supported

### Run Container
docker run --env-file .env multisource-rag


The application is CLI-based, so the container executes and exits after producing the response.
- Explicit workflow control using LangGraph
- Designed with production edge cases in mind

Project Status

End-to-end RAG pipeline implemented

Multi-source retrieval supported

Explicit workflow control using LangGraph

Dockerized for reproducible execution

Designed with production edge cases in mind

