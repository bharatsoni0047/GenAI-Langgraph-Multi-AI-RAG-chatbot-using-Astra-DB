# Multi-Source RAG Chatbot using LangGraph & AstraDB

## Problem Statement

Traditional chatbots usually fall into one of two categories:

- They rely only on static internal documents, which limits their knowledge scope.
- They depend entirely on external sources, which removes domain control and reliability.

This project addresses these limitations by building a **multi-source Retrieval-Augmented Generation (RAG) system** that can intelligently decide:

- when to use **internal knowledge** stored in a vector database (AstraDB)
- when to fall back to **external public knowledge** (Wikipedia)

All routing and execution logic is orchestrated using **LangGraph** to enable explicit, production-style workflow control.

---

## Project Overview

This project implements a **LangGraph-based RAG chatbot** that performs the following steps:

1. Accepts a user query
2. Routes the query to the appropriate data source
3. Retrieves relevant contextual information
4. Generates a grounded response using an LLM

The system is designed to handle real-world scenarios such as:

- multiple retrieval sources
- heterogeneous response formats
- LLM compatibility constraints

---

## High-Level Architecture (Flow Graph)

     ┌──────────────┐
        │   User Query │
        └──────┬───────┘
               │
               ▼
      ┌─────────────────┐
      │  Router Node     │
      │ (LLM-based JSON) │
      └──────┬──────────┘
    ┌─────────┴─────────┐
    ▼                   ▼
┌──────────────┐ ┌──────────────┐
│ AstraDB │ │ Wikipedia │
│ Vector Store │ │ Search Tool │
└──────┬───────┘ └──────┬───────┘
▼ ▼
┌─────────────────┐
│ Generation Node │
│ (LLM Answer) │
└──────┬──────────┘
▼
┌──────────────┐
│ Final Answer │
└──────────────┘




---

## Key Components

### 1. LLM-Based Router (LangGraph)

- Uses an LLM to classify incoming queries
- Routes queries to:
  - AstraDB vector store for domain-specific content
  - Wikipedia for general knowledge
- Implemented using **strict JSON-based routing** to avoid tool-calling dependencies

LangGraph conditional edges are used to control execution flow based on the router’s output.

---

### 2. AstraDB Vector Store (Internal Knowledge)

- Source documents are:
  - loaded from web sources
  - split into chunks using recursive text splitting
  - embedded using HuggingFace MiniLM embeddings
- Embeddings are stored in **AstraDB (Cassandra Vector Search)**

This source is used for controlled, domain-specific retrieval.

---

### 3. Wikipedia Tool (External Knowledge)

- Acts as a fallback for general or public knowledge queries
- Implemented as a separate LangGraph node
- Enables hybrid retrieval across internal and external sources

---

### 4. Generation Node (RAG Completion)

- Retrieved documents are normalized into a unified format
- Context is constructed from retrieved content
- An LLM generates the final response grounded in retrieved data

This ensures the system functions as a complete RAG pipeline rather than a retrieval-only demo.

---

### 5. Production Edge-Case Handling

The system handles different retrieval output formats, including:

- `(Document, score)` tuples from vector databases
- `Document` objects
- raw text responses from external tools

All outputs are normalized before being passed to the generation stage to avoid runtime failures.

---

## Tech Stack

- **LangGraph** – Workflow orchestration and conditional routing
- **LangChain** – RAG abstractions and utilities
- **AstraDB (Cassandra Vector DB)** – Vector storage and semantic retrieval
- **Groq LLM** – LLM inference for routing and generation
- **HuggingFace Embeddings** – Text embedding generation
- **Wikipedia API** – External knowledge source
- **Python** – Core implementation language

---

## Possible Extensions

- Add citation tracking for generated answers
- Introduce retry and fallback logic in LangGraph
- Stream token-level responses
- Integrate additional tools (Arxiv, SQL, external APIs)

---

## Project Status

- End-to-end RAG pipeline implemented
- Multi-source retrieval supported
- Explicit workflow control using LangGraph
- Designed with production edge cases in mind
