from pathlib import Path
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from vector_db import VectorDB
from document_db import DocumentDB
from ollama_client import OllamaClient
from vector_core import get_distance_function


# ============================================================
# Application
# ============================================================

app = FastAPI(
    title="NeuroVector",
    description="Vector search and RAG engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# Ollama
# ============================================================

ollama = OllamaClient()


# ============================================================
# Demo Vector Database
# ============================================================

DEMO_DIMENSIONS = 16

vector_db = VectorDB(
    dimensions=DEMO_DIMENSIONS,
    metric="cosine"
)


# ============================================================
# Document Database
# ============================================================

document_db = DocumentDB(
    ollama=ollama
)


# ============================================================
# Request Models
# ============================================================

class InsertRequest(BaseModel):
    metadata: str
    category: str
    embedding: list[float]


class DocumentInsertRequest(BaseModel):
    title: str
    text: str


class DocumentSearchRequest(BaseModel):
    question: str
    k: int = 5


class AskRequest(BaseModel):
    question: str
    k: int = 5


# ============================================================
# Demo Data
# ============================================================

DEMO_DATA = [
    ("Python Programming", "cs"),
    ("Machine Learning", "cs"),
    ("Data Structures", "cs"),
    ("Algorithms", "cs"),
    ("Artificial Intelligence", "cs"),
    ("Neural Networks", "cs"),

    ("Linear Algebra", "math"),
    ("Calculus", "math"),
    ("Probability", "math"),
    ("Statistics", "math"),

    ("Pizza", "food"),
    ("Burger", "food"),
    ("Pasta", "food"),
    ("Sushi", "food"),

    ("Formula One", "sports"),
    ("Football", "sports"),
    ("Basketball", "sports"),
    ("Tennis", "sports"),
    ("Cricket", "sports"),
    ("MotoGP", "sports")
]


def create_demo_embedding(
    index: int,
    category: str
) -> list[float]:

    category_offsets = {
        "cs": 0.0,
        "math": 1.0,
        "food": 2.0,
        "sports": 3.0,
        "doc": 4.0
    }

    offset = category_offsets.get(
        category,
        0.0
    )

    base = float(index) * 0.015

    embedding = []

    for i in range(16):
        value = (
            0.1
            + offset
            + base
            + ((i % 4) * 0.02)
        )

        embedding.append(value)

    return embedding


def initialize_demo_data() -> None:

    if vector_db.size() > 0:
        return

    for index, (metadata, category) in enumerate(
        DEMO_DATA,
        start=1
    ):

        vector_db.insert(
            metadata=metadata,
            category=category,
            embedding=create_demo_embedding(
                index,
                category
            ),
            item_id=index
        )


initialize_demo_data()


# ============================================================
# Helpers
# ============================================================

def set_metric(metric: str) -> None:

    metric = metric.lower()

    if metric not in (
        "cosine",
        "euclidean",
        "manhattan"
    ):
        metric = "cosine"

    distance_function = get_distance_function(
        metric
    )

    vector_db.metric = metric
    vector_db.distance_function = distance_function

    vector_db.brute_force.distance_function = (
        distance_function
    )

    vector_db.kd_tree.distance_function = (
        distance_function
    )

    vector_db.hnsw.distance_function = (
        distance_function
    )


def serialize_item(
    item,
    distance=None
) -> dict:

    result = {
        "id": item.id,
        "metadata": item.metadata,
        "category": item.category,
        "embedding": item.emb
    }

    if distance is not None:
        result["distance"] = distance

    return result


# ============================================================
# Root / Frontend
# ============================================================

@app.get("/")
def root():

    index_path = (
        Path(__file__).resolve().parent.parent
        / "index.html"
    )

    if index_path.exists():
        return FileResponse(index_path)

    return {
        "name": "NeuroVector",
        "status": "running"
    }


# ============================================================
# Demo Query → 16D Vector
# ============================================================

def create_query_embedding(
    query: str
) -> list[float]:

    text = query.lower()

    category = "cs"

    if any(
        word in text
        for word in [
            "math",
            "calculus",
            "probability",
            "statistics",
            "algebra",
            "matrix"
        ]
    ):
        category = "math"

    elif any(
        word in text
        for word in [
            "food",
            "pizza",
            "burger",
            "pasta",
            "sushi",
            "recipe"
        ]
    ):
        category = "food"

    elif any(
        word in text
        for word in [
            "sport",
            "football",
            "basketball",
            "tennis",
            "cricket",
            "f1",
            "formula",
            "motogp",
            "game"
        ]
    ):
        category = "sports"

    category_offsets = {
        "cs": 0.0,
        "math": 1.0,
        "food": 2.0,
        "sports": 3.0
    }

    offset = category_offsets[category]

    seed = sum(
        ord(character)
        for character in text
    )

    base = (
        (seed % 100) / 1000.0
    )

    return [
        0.1
        + offset
        + base
        + ((i % 4) * 0.02)
        for i in range(16)
    ]


# ============================================================
# GET /items
# ============================================================

@app.get("/items")
def get_items():

    # Frontend expects the array directly.
    return [
        serialize_item(item)
        for item in vector_db.get_items()
    ]


# ============================================================
# GET /search
# ============================================================

@app.get("/search")
def search(
    v: str = "",
    q: str = "",
    k: int = 5,
    algo: str = "hnsw",
    metric: str = "cosine"
):

    try:

        # The frontend sends the embedding as:
        # ?v=0.1,0.2,0.3,...
        if v:

            query = [
                float(value)
                for value in v.split(",")
            ]

        elif q:

            query = create_query_embedding(q)

        else:

            raise HTTPException(
                status_code=400,
                detail="Query vector is required."
            )

        if len(query) != DEMO_DIMENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Expected {DEMO_DIMENSIONS} "
                    f"dimensions."
                )
            )

        set_metric(metric)

        start = time.perf_counter()

        results = vector_db.search(
            query=query,
            k=k,
            algorithm=algo
        )

        latency_us = (
            time.perf_counter() - start
        ) * 1_000_000

        return {
            "results": [
                serialize_item(
                    item,
                    distance
                )
                for item, distance in results
            ],
            "latencyUs": round(
                latency_us,
                2
            )
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


# ============================================================
# POST /insert
# ============================================================

@app.post("/insert")
def insert_vector(
    request: InsertRequest
):

    try:

        item = vector_db.insert(
            metadata=request.metadata,
            category=request.category,
            embedding=request.embedding
        )

        return serialize_item(item)

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


# ============================================================
# DELETE /delete/{id}
# ============================================================

@app.delete("/delete/{item_id}")
def delete_vector(
    item_id: int
):

    if not vector_db.remove(item_id):

        raise HTTPException(
            status_code=404,
            detail="Vector not found."
        )

    return {
        "success": True,
        "id": item_id
    }


# ============================================================
# GET /benchmark
# ============================================================

@app.get("/benchmark")
def benchmark(
    v: str = "",
    q: str = "",
    k: int = 5,
    metric: str = "cosine"
):

    if v:

        query = [
            float(value)
            for value in v.split(",")
        ]

    elif q:

        query = create_query_embedding(q)

    else:

        query = create_query_embedding(
            "binary tree algorithm"
        )

    set_metric(metric)

    result = vector_db.benchmark(
        query=query,
        k=k
    )

    return {
        "bruteforceUs": round(
            result["brute_force"]["time_ms"] * 1000,
            2
        ),
        "kdtreeUs": round(
            result["kd_tree"]["time_ms"] * 1000,
            2
        ),
        "hnswUs": round(
            result["hnsw"]["time_ms"] * 1000,
            2
        )
    }


# ============================================================
# GET /hnsw-info
# ============================================================

@app.get("/hnsw-info")
def hnsw_info():

    return vector_db.hnsw_info()


# ============================================================
# GET /stats
# ============================================================

@app.get("/stats")
def stats():

    return vector_db.stats()


# ============================================================
# GET /status
# ============================================================

@app.get("/status")
def status():

    ollama_available = ollama.is_available()

    return {
        "ollamaAvailable": ollama_available,
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "docDims": document_db.dimensions,
        "docCount": len(
            document_db.documents
        )
    }


# ============================================================
# POST /doc/insert
# ============================================================

@app.post("/doc/insert")
def insert_document(
    request: DocumentInsertRequest
):

    try:

        document = document_db.insert_document(
            name=request.title,
            text=request.text
        )

        return {
            "id": document["id"],
            "title": request.title,
            "chunks": document["chunkCount"],
            "dims": document_db.dimensions
        }

    except Exception as error:

        return {
            "error": str(error)
        }


# ============================================================
# GET /doc/list
# ============================================================

@app.get("/doc/list")
def list_documents():

    documents = []

    for document in document_db.documents.values():

        document_id = document["id"]

        chunk_ids = document.get(
            "chunkIds",
            []
        )

        first_chunk = None

        if chunk_ids:

            first_chunk = document_db.chunks.get(
                chunk_ids[0]
            )

        preview = ""

        if first_chunk:

            preview = first_chunk.text[:180]

        full_word_count = 0

        for chunk_id in chunk_ids:

            chunk = document_db.chunks.get(
                chunk_id
            )

            if chunk:
                full_word_count += len(
                    chunk.text.split()
                )

        documents.append(
            {
                "id": document_id,
                "title": document["name"],
                "preview": preview,
                "words": full_word_count
            }
        )

    return documents


# ============================================================
# DELETE /doc/delete/{id}
# ============================================================

@app.delete("/doc/delete/{document_id}")
def delete_document(
    document_id: int
):

    deleted = document_db.delete_document(
        document_id
    )

    if not deleted:

        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    return {
        "success": True,
        "id": document_id
    }


# ============================================================
# POST /doc/search
# ============================================================

@app.post("/doc/search")
def document_search(
    request: DocumentSearchRequest
):

    try:

        results = document_db.search(
            query=request.question,
            k=request.k
        )

        contexts = []

        for result in results:

            contexts.append(
                {
                    "title": result["documentName"],
                    "text": result["text"],
                    "distance": result["distance"]
                }
            )

        return {
            "contexts": contexts
        }

    except Exception as error:

        return {
            "contexts": [],
            "error": str(error)
        }


# ============================================================
# POST /doc/ask
# ============================================================

@app.post("/doc/ask")
def ask_ai(
    request: AskRequest
):

    try:

        result = document_db.ask(
            question=request.question,
            k=request.k
        )

        contexts = []

        for source in result.get(
            "sources",
            []
        ):

            contexts.append(
                {
                    "title": source["documentName"],
                    "text": source["text"],
                    "distance": source["distance"]
                }
            )

        return {
            "answer": result["answer"],
            "contexts": contexts,
            "model": ollama.gen_model
        }

    except Exception as error:

        return {
            "error": str(error),
            "answer": "",
            "contexts": [],
            "model": ollama.gen_model
        }