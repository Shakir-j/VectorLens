# VectorLens — Vector Search & RAG Engine

A Python-based vector search and Retrieval-Augmented Generation (RAG) engine that explores how vector databases, similarity search, embeddings, and local LLMs work under the hood.

VectorLens implements HNSW, KD-Tree, and Brute Force search algorithms, supports Cosine, Euclidean, and Manhattan distance metrics, provides an interactive 2D PCA visualization, and includes a document-based RAG pipeline powered by local AI models through Ollama.

---

## Features

- **HNSW Search** — Approximate nearest-neighbor search using a hierarchical graph
- **KD-Tree Search** — Multidimensional spatial search using recursive partitioning
- **Brute Force Search** — Exact nearest-neighbor search used as a baseline
- **3 Distance Metrics**
  - Cosine Distance
  - Euclidean Distance
  - Manhattan Distance
- **20 Preloaded Demo Vectors** across Computer Science, Mathematics, Food, and Sports
- **16-Dimensional Demo Vector Space**
- **2D PCA Visualization** of the vector space
- **Real Document Embeddings** using Ollama's `nomic-embed-text`
- **768-Dimensional Document Embeddings**
- **Document Chunking** with 250-word chunks and 30-word overlap
- **Semantic Document Search**
- **RAG Pipeline** using HNSW retrieval and Ollama's `llama3.2`
- **Interactive Web Interface**
- **FastAPI REST API**
- **Algorithm Benchmarking**
- **HNSW Graph Information**
- **Fully Local AI Processing** through Ollama

---

## How It Works

VectorLens has two main parts:

1. A demo vector database for exploring vector search algorithms.
2. A document-based RAG system for semantic search and AI question answering.

### Vector Search

```text
User Query
    |
    v
16D Query Vector
    |
    +----------------+----------------+
    |                |                |
    v                v                v
  HNSW            KD-Tree        Brute Force
    |                |                |
    +----------------+----------------+
                     |
                     v
             Distance Calculation
                     |
                     v
              Nearest Neighbors
                     |
                     v
                Web Interface
```

The demo-vector system uses 16-dimensional vectors so that the vector space can also be visualized using PCA.

---

## RAG Pipeline

```text
Document
    |
    v
Document Chunking
    |
    v
nomic-embed-text
    |
    v
768-Dimensional Embedding
    |
    v
Vector Index
    |
    v
Semantic Search
    |
    v
Relevant Document Chunks
    |
    v
llama3.2
    |
    v
Generated Answer
```

When a question is asked, the question is converted into an embedding and compared against stored document embeddings. The most relevant chunks are retrieved and supplied to the local language model as context.

---

# HNSW

**HNSW (Hierarchical Navigable Small World)** is a graph-based approximate nearest-neighbor search algorithm.

VectorLens implements the HNSW data structure directly in Python rather than relying on an external HNSW library for the core search implementation.

The graph is organized into multiple layers:

```text
Higher Layer       O---------O

Middle Layer     O---O-----O---O

Base Layer     O-O-O-O-O-O-O-O-O-O
```

Higher layers contain fewer nodes and provide long-range connections.

Search starts from an upper layer and moves through the graph toward increasingly closer candidates. The search then continues at lower layers until the nearest candidates are found.

HNSW is particularly useful for approximate nearest-neighbor search on large vector collections.

---

# KD-Tree

A **KD-Tree (K-Dimensional Tree)** is a tree-based data structure for organizing points in multidimensional space.

The implementation recursively partitions the vector space along different dimensions.

For example:

```text
Dimension 0
    |
    +--- Left Subtree
    |
    +--- Right Subtree
             |
             v
        Dimension 1
```

During search, the tree can eliminate regions that cannot contain better candidates.

KD-Trees are useful for lower-dimensional data but generally become less effective as dimensionality increases because of the curse of dimensionality.

---

# Brute Force

Brute Force performs an exact nearest-neighbor search.

For every query vector, the distance to every stored vector is calculated:

```text
Query
  |
  +--- Distance -> Vector 1
  +--- Distance -> Vector 2
  +--- Distance -> Vector 3
  +--- ...
  +--- Distance -> Vector N
```

The vectors are then sorted by distance and the closest `k` results are returned.

Although this approach becomes expensive for large datasets, it provides an exact baseline for evaluating approximate search algorithms such as HNSW.

---

# Distance Metrics

VectorLens supports three distance metrics.

## Cosine Distance

Cosine distance measures the angular difference between two vectors.

```text
Cosine Distance = 1 - Cosine Similarity
```

Smaller values indicate greater similarity.

Cosine distance is commonly used with text embeddings because the direction of a vector can be more important than its magnitude.

## Euclidean Distance

Euclidean distance represents the straight-line distance between two vectors.

```text
d(a,b) = sqrt(Σ(ai - bi)²)
```

## Manhattan Distance

Manhattan distance is the sum of the absolute differences between vector coordinates.

```text
d(a,b) = Σ|ai - bi|
```

The distance metric can be selected from the web interface.

---

# Demo Vector Search

The application starts with **20 preloaded 16-dimensional vectors**.

The vectors represent several categories:

- Computer Science
- Mathematics
- Food
- Sports

Example search concepts include:

```text
binary tree
sushi
basketball
calculus
```

The interface allows the user to select:

### Search Algorithm

- HNSW
- KD-Tree
- Brute Force

### Distance Metric

- Cosine
- Euclidean
- Manhattan

The application returns the nearest vectors along with their distance values.

---

# PCA Visualization

The demo vectors exist in a 16-dimensional space.

Because 16 dimensions cannot be directly visualized on a normal 2D screen, VectorLens uses **Principal Component Analysis (PCA)** to project the vectors into two dimensions.

```text
16D Vector Space
       |
       v
      PCA
       |
       v
2D Projection
       |
       v
Interactive Visualization
```

The visualization makes it easier to understand how vectors from different semantic categories are positioned relative to each other.

---

# Document Embeddings

VectorLens supports real document embeddings using Ollama.

Documents are sent to:

```text
Ollama
   |
   v
nomic-embed-text
   |
   v
768-Dimensional Vector
```

The embedding model converts text into a numerical representation that captures semantic information.

This allows the system to perform semantic search instead of relying only on exact keyword matching.

For example, a document containing:

```text
Machine learning allows systems to learn patterns from data.
```

can potentially be retrieved for a question such as:

```text
How do computers learn from examples?
```

even though the exact words may differ.

---

# Document Chunking

Long documents are divided into smaller overlapping chunks.

The current chunking strategy uses:

- **250 words per chunk**
- **30-word overlap**

Conceptually:

```text
Chunk 1
[--------------------------------]

Chunk 2
              [--------------------------------]

Chunk 3
                            [--------------------------------]
```

The overlap helps preserve context when information crosses chunk boundaries.

Each chunk is embedded separately and stored for semantic retrieval.

---

# Retrieval-Augmented Generation

The RAG system combines semantic retrieval with a local language model.

When the user asks a question:

### 1. Question Embedding

```text
Question
   |
   v
nomic-embed-text
   |
   v
768D Query Vector
```

### 2. Semantic Retrieval

The query vector is searched against the stored document chunk vectors.

The most relevant chunks are retrieved.

### 3. Context Construction

The retrieved chunks are supplied as context to the language model.

### 4. Answer Generation

```text
User Question
      +
Retrieved Context
      |
      v
   llama3.2
      |
      v
Generated Answer
```

The system also returns the retrieved document contexts used during the answer generation process.

---

# Prerequisites

You need:

- Python 3.10+
- Git
- Ollama

No C++ compiler is required.

---

# Installation

## 1. Clone the Repository

```powershell
git clone https://github.com/Shakir-j/VectorLens.git
cd VectorLens
```

## 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

The project uses Python packages including:

- FastAPI
- Uvicorn
- NumPy
- Requests
- scikit-learn

---

# Ollama Setup

Install Ollama and make sure it is running.

Pull the embedding model:

```powershell
ollama pull nomic-embed-text
```

Pull the generation model:

```powershell
ollama pull llama3.2
```

Verify the installed models:

```powershell
ollama list
```

The application uses:

```text
nomic-embed-text
```

for embeddings and:

```text
llama3.2
```

for RAG answer generation.

---

# Running VectorLens

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start the application:

```powershell
python run.py
```

The application runs at:

```text
http://localhost:8080
```

The Python launcher automatically opens the browser.

---

# Using the Application

## Search Demo Vectors

Enter a concept such as:

```text
binary tree
```

Choose an algorithm:

```text
HNSW
KD-Tree
Brute Force
```

Choose a distance metric:

```text
Cosine
Euclidean
Manhattan
```

Then run the search.

The interface displays the nearest vectors and their distances.

---

## Compare Search Algorithms

The benchmark functionality allows the three search approaches to be compared using the same query.

```text
HNSW
KD-Tree
Brute Force
```

Brute Force provides an exact baseline while HNSW provides approximate graph-based search.

---

## Insert a Document

A document can be inserted by providing:

- Title
- Text

The system then:

1. Splits the document into chunks.
2. Generates embeddings using `nomic-embed-text`.
3. Creates 768-dimensional vectors.
4. Stores the vectors in the document index.

---

## Semantic Document Search

A natural-language question can be entered into the document search interface.

The question is embedded using `nomic-embed-text` and compared with the stored document chunk embeddings.

The most relevant contexts are returned.

---

## Ask AI

After documents have been inserted, questions can be asked through the AI interface.

The complete flow is:

```text
Question
   |
   v
Embedding
   |
   v
Vector Search
   |
   v
Relevant Chunks
   |
   v
llama3.2
   |
   v
Answer + Sources
```

---

# REST API

The backend is implemented using **FastAPI**.

Base URL:

```text
http://localhost:8080
```

## Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/search` | Search for nearest vectors |
| POST | `/insert` | Insert a vector |
| DELETE | `/delete/{id}` | Delete a vector |
| GET | `/items` | List vectors |
| GET | `/benchmark` | Compare search algorithms |
| GET | `/hnsw-info` | Get HNSW graph information |
| GET | `/stats` | Get vector database statistics |

## Document and RAG Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/doc/insert` | Insert and embed a document |
| GET | `/doc/list` | List documents |
| DELETE | `/doc/delete/{id}` | Delete a document |
| POST | `/doc/search` | Perform semantic document search |
| POST | `/doc/ask` | Retrieve context and generate an answer |
| GET | `/status` | Check Ollama and system status |

---

# Project Structure

```text
VectorLens/
│
├── app/
│   ├── api.py
│   ├── vector_core.py
│   ├── brute_force.py
│   ├── kd_tree.py
│   ├── hnsw.py
│   ├── vector_db.py
│   ├── document_db.py
│   ├── chunker.py
│   └── ollama_client.py
│
├── index.html
├── run.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Component Overview

### `vector_core.py`

Contains the core vector representation and distance metric implementations.

### `brute_force.py`

Implements exact nearest-neighbor search by comparing the query against every stored vector.

### `kd_tree.py`

Implements multidimensional tree-based vector organization and search.

### `hnsw.py`

Implements the hierarchical graph structure used for approximate nearest-neighbor search.

### `vector_db.py`

Provides a unified vector database interface and manages the different search algorithms.

### `chunker.py`

Splits documents into overlapping chunks.

### `document_db.py`

Handles document storage, embeddings, semantic retrieval, and the RAG workflow.

### `ollama_client.py`

Communicates with the local Ollama server for:

- Text embeddings
- LLM generation
- Model availability

### `api.py`

Provides the FastAPI backend and exposes the vector search, document, benchmark, and RAG endpoints.

### `run.py`

Starts the Uvicorn/FastAPI server and automatically opens the web interface.

---

# Algorithm Comparison

| Algorithm | Type | Main Idea | Purpose |
|---|---|---|---|
| **Brute Force** | Exact | Compare with every vector | Correctness baseline |
| **KD-Tree** | Tree-based | Partition space by dimensions | Efficient lower-dimensional search |
| **HNSW** | Approximate | Navigate a multilayer graph | Efficient nearest-neighbor search |

The three implementations allow the project to demonstrate different approaches to nearest-neighbor retrieval.

---

# Why Vector Search?

Traditional keyword search looks for matching words.

For example:

```text
"machine learning"
```

A semantic vector search system instead represents text as vectors and searches for nearby representations in vector space.

```text
Text
  |
  v
Embedding
  |
  v
Vector
  |
  v
Similarity Search
  |
  v
Relevant Information
```

This makes vector search useful for semantic retrieval, recommendation systems, document search, and RAG applications.

---

# Technology Stack

### Backend

- Python
- FastAPI
- Uvicorn

### Vector Search

- HNSW
- KD-Tree
- Brute Force
- NumPy

### AI / ML

- Ollama
- `nomic-embed-text`
- `llama3.2`
- PCA
- scikit-learn

### Frontend

- HTML
- CSS
- JavaScript

---

# Future Improvements

Possible extensions include:

- Persistent vector storage
- Larger document collections
- Additional embedding models
- HNSW parameter tuning
- Metadata filtering
- Hybrid keyword + vector search
- Additional document loaders
- Larger-scale benchmarking
- GPU-accelerated inference
- API authentication

---

# License

MIT
