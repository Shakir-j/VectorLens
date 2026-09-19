import requests


class OllamaClient:
    """
    Client for the local Ollama API.

    Default models match the original project:
    - nomic-embed-text for embeddings
    - llama3.2 for text generation
    """

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        embed_model: str = "nomic-embed-text",
        gen_model: str = "llama3.2"
    ):
        self.host = host.rstrip("/")
        self.embed_model = embed_model
        self.gen_model = gen_model

    # --------------------------------------------------
    # Check Ollama
    # --------------------------------------------------

    def is_available(self) -> bool:
        try:
            response = requests.get(
                f"{self.host}/api/tags",
                timeout=5
            )

            return response.ok

        except requests.RequestException:
            return False

    # --------------------------------------------------
    # Get installed models
    # --------------------------------------------------

    def get_models(self) -> list[str]:

        response = requests.get(
            f"{self.host}/api/tags",
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        return [
            model.get("name", "")
            for model in data.get("models", [])
        ]

    # --------------------------------------------------
    # Generate embedding
    # --------------------------------------------------

    def embed(self, text: str) -> list[float]:

        response = requests.post(
            f"{self.host}/api/embeddings",
            json={
                "model": self.embed_model,
                "prompt": text
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        embedding = data.get("embedding")

        if not embedding:
            raise RuntimeError(
                "Ollama returned no embedding."
            )

        return [
            float(value)
            for value in embedding
        ]

    # --------------------------------------------------
    # Generate text
    # --------------------------------------------------

    def generate(self, prompt: str) -> str:

        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.gen_model,
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )

        response.raise_for_status()

        data = response.json()

        return data.get("response", "")

    # --------------------------------------------------
    # Model status
    # --------------------------------------------------

    def status(self) -> dict:

        available = self.is_available()

        result = {
            "available": available,
            "host": self.host,
            "embeddingModel": self.embed_model,
            "generationModel": self.gen_model
        }

        if available:
            try:
                result["models"] = self.get_models()
            except requests.RequestException:
                result["models"] = []

        else:
            result["models"] = []

        return result