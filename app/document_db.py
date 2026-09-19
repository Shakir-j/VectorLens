from dataclasses import dataclass

from chunker import chunk_text
from hnsw import HNSW
from brute_force import BruteForce
from ollama_client import OllamaClient
from vector_core import VectorItem


@dataclass
class DocumentChunk:
    id: int
    document_id: int
    document_name: str
    text: str


class DocumentDB:
    """
    Document storage and RAG search layer.

    Pipeline:

        Document
            ↓
        Text chunks
            ↓
        Ollama embeddings
            ↓
        Vector indexes
            ↓
        Semantic search
            ↓
        Relevant context
            ↓
        Ollama LLM
    """

    def __init__(
        self,
        ollama: OllamaClient | None = None
    ):
        self.ollama = ollama or OllamaClient()

        self.documents: dict[int, dict] = {}
        self.chunks: dict[int, DocumentChunk] = {}

        self.next_document_id = 1
        self.next_chunk_id = 1

        self.dimensions: int | None = None

        self.hnsw: HNSW | None = None
        self.brute_force: BruteForce | None = None

    # --------------------------------------------------
    # Initialize vector indexes
    # --------------------------------------------------

    def _initialize_indexes(
        self,
        dimensions: int
    ) -> None:

        if self.dimensions is not None:
            return

        self.dimensions = dimensions

        self.hnsw = HNSW(
            m=16,
            ef_build=200,
            metric="cosine"
        )

        self.brute_force = BruteForce(
            metric="cosine"
        )

    # --------------------------------------------------
    # Insert document
    # --------------------------------------------------

    def insert_document(
        self,
        name: str,
        text: str
    ) -> dict:

        if not text.strip():
            raise ValueError(
                "Document text cannot be empty."
            )

        document_id = self.next_document_id
        self.next_document_id += 1

        chunks = chunk_text(
            text,
            chunk_size=250,
            overlap=30
        )

        if not chunks:
            raise ValueError(
                "Document produced no chunks."
            )

        document = {
            "id": document_id,
            "name": name,
            "chunkCount": len(chunks)
        }

        self.documents[document_id] = document

        inserted_chunks = []

        for chunk in chunks:

            embedding = self.ollama.embed(
                chunk
            )

            self._initialize_indexes(
                len(embedding)
            )

            chunk_id = self.next_chunk_id
            self.next_chunk_id += 1

            document_chunk = DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                document_name=name,
                text=chunk
            )

            self.chunks[chunk_id] = document_chunk

            vector_item = VectorItem(
                id=chunk_id,
                metadata=chunk,
                category=name,
                emb=embedding
            )

            self.hnsw.insert(vector_item)
            self.brute_force.insert(vector_item)

            inserted_chunks.append(
                chunk_id
            )

        document["chunkIds"] = inserted_chunks

        return document

    # --------------------------------------------------
    # List documents
    # --------------------------------------------------

    def list_documents(self) -> list[dict]:

        return list(
            self.documents.values()
        )

    # --------------------------------------------------
    # Delete document
    # --------------------------------------------------

    def delete_document(
        self,
        document_id: int
    ) -> bool:

        if document_id not in self.documents:
            return False

        document = self.documents[
            document_id
        ]

        chunk_ids = document.get(
            "chunkIds",
            []
        )

        for chunk_id in chunk_ids:

            if self.hnsw is not None:
                self.hnsw.remove(
                    chunk_id
                )

            if self.brute_force is not None:
                self.brute_force.remove(
                    chunk_id
                )

            self.chunks.pop(
                chunk_id,
                None
            )

        del self.documents[
            document_id
        ]

        return True

    # --------------------------------------------------
    # Search documents
    # --------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        max_distance: float = 0.7
    ) -> list[dict]:

        if not query.strip():
            return []

        if self.dimensions is None:
            return []

        query_embedding = self.ollama.embed(
            query
        )

        # The original project uses brute force
        # for small document collections and HNSW
        # once the collection becomes larger.
        if (
            self.brute_force is not None
            and self.brute_force.size() < 10
        ):

            results = self.brute_force.search(
                query_embedding,
                k
            )

            raw_results = [
                (
                    result.item,
                    result.distance
                )
                for result in results
            ]

        else:

            raw_results = self.hnsw.search(
                query_embedding,
                k=k,
                ef=50
            )

        output = []

        for item, distance in raw_results:

            if distance > max_distance:
                continue

            chunk = self.chunks.get(
                item.id
            )

            if chunk is None:
                continue

            output.append(
                {
                    "id": chunk.id,
                    "documentId": chunk.document_id,
                    "documentName": chunk.document_name,
                    "text": chunk.text,
                    "distance": distance
                }
            )

        return output

    # --------------------------------------------------
    # Ask AI / RAG
    # --------------------------------------------------

    def ask(
        self,
        question: str,
        k: int = 5
    ) -> dict:

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        results = self.search(
            question,
            k=k
        )

        if not results:
            return {
                "answer": (
                    "I could not find relevant "
                    "information in the documents."
                ),
                "sources": []
            }

        context_parts = []

        for index, result in enumerate(
            results,
            start=1
        ):

            context_parts.append(
                f"[Source {index}]\n"
                f"Document: {result['documentName']}\n"
                f"{result['text']}"
            )

        context = "\n\n".join(
            context_parts
        )

        prompt = f"""
You are a helpful AI assistant.

Answer the user's question using only
the provided document context.

If the answer cannot be determined from
the context, say that the information is
not available in the provided documents.

Be concise and accurate.

Document context:

{context}

User question:

{question}

Answer:
""".strip()

        answer = self.ollama.generate(
            prompt
        )

        return {
            "answer": answer,
            "sources": results
        }

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    def stats(self) -> dict:

        return {
            "documentCount": len(
                self.documents
            ),
            "chunkCount": len(
                self.chunks
            ),
            "dimensions": self.dimensions
        }