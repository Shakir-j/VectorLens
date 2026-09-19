import time
import threading

from vector_core import VectorItem, get_distance_function
from brute_force import BruteForce
from kd_tree import KDTree
from hnsw import HNSW


class VectorDB:
    """
    Main vector database.

    Maintains three search indexes:
    - Brute Force
    - KD-Tree
    - HNSW

    This mirrors the architecture of the original C++ project.
    """

    def __init__(
        self,
        dimensions: int,
        metric: str = "euclidean",
        hnsw_m: int = 16,
        hnsw_ef_build: int = 200
    ):
        self.dimensions = dimensions
        self.metric = metric

        self.items: dict[int, VectorItem] = {}
        self.next_id = 1

        self.lock = threading.RLock()

        self.brute_force = BruteForce(metric=metric)

        self.kd_tree = KDTree(
            dimensions=dimensions,
            metric=metric
        )

        self.hnsw = HNSW(
            m=hnsw_m,
            ef_build=hnsw_ef_build,
            metric=metric
        )

        self.distance_function = get_distance_function(metric)

    # --------------------------------------------------
    # Insert
    # --------------------------------------------------

    def insert(
        self,
        metadata: str,
        category: str,
        embedding: list[float],
        item_id: int | None = None
    ) -> VectorItem:

        if len(embedding) != self.dimensions:
            raise ValueError(
                f"Expected {self.dimensions} dimensions, "
                f"got {len(embedding)}"
            )

        with self.lock:

            if item_id is None:
                item_id = self.next_id
                self.next_id += 1
            else:
                self.next_id = max(
                    self.next_id,
                    item_id + 1
                )

            item = VectorItem(
                id=item_id,
                metadata=metadata,
                category=category,
                emb=embedding
            )

            self.items[item_id] = item

            self.brute_force.insert(item)
            self.kd_tree.insert(item)
            self.hnsw.insert(item)

            return item

    # --------------------------------------------------
    # Remove
    # --------------------------------------------------

    def remove(self, item_id: int) -> bool:

        with self.lock:

            if item_id not in self.items:
                return False

            item = self.items[item_id]

            self.brute_force.remove(item_id)
            self.hnsw.remove(item_id)

            del self.items[item_id]

            # The original implementation rebuilds the KD-Tree
            # after deletion.
            self._rebuild_kd_tree()

            return True

    # --------------------------------------------------
    # Rebuild KD-Tree
    # --------------------------------------------------

    def _rebuild_kd_tree(self) -> None:

        self.kd_tree = KDTree(
            dimensions=self.dimensions,
            metric=self.metric
        )

        for item in self.items.values():
            self.kd_tree.insert(item)

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    def search(
        self,
        query: list[float],
        k: int = 5,
        algorithm: str = "hnsw"
    ) -> list[tuple[VectorItem, float]]:

        if len(query) != self.dimensions:
            raise ValueError(
                f"Expected {self.dimensions} dimensions, "
                f"got {len(query)}"
            )

        with self.lock:

            algorithm = algorithm.lower()

            if algorithm in ("brute", "bruteforce", "brute-force"):
                results = self.brute_force.search(
                    query,
                    k
                )

                return [
                    (result.item, result.distance)
                    for result in results
                ]

            if algorithm in ("kd", "kdtree", "kd-tree"):
                return self.kd_tree.search(
                    query,
                    k
                )

            # Default: HNSW
            return self.hnsw.search(
                query,
                k=k,
                ef=max(50, k)
            )

    # --------------------------------------------------
    # Get all items
    # --------------------------------------------------

    def get_items(self) -> list[VectorItem]:

        with self.lock:
            return list(self.items.values())

    # --------------------------------------------------
    # Size
    # --------------------------------------------------

    def size(self) -> int:

        with self.lock:
            return len(self.items)

    # --------------------------------------------------
    # Benchmark
    # --------------------------------------------------

    def benchmark(
        self,
        query: list[float],
        k: int = 5
    ) -> dict:

        if len(query) != self.dimensions:
            raise ValueError(
                f"Expected {self.dimensions} dimensions, "
                f"got {len(query)}"
            )

        benchmark_results = {}

        # -----------------------------
        # Brute Force
        # -----------------------------

        start = time.perf_counter()

        brute_results = self.brute_force.search(
            query,
            k
        )

        brute_time = (
            time.perf_counter() - start
        ) * 1000

        benchmark_results["brute_force"] = {
            "time_ms": brute_time,
            "results": [
                {
                    "id": result.item.id,
                    "distance": result.distance
                }
                for result in brute_results
            ]
        }

        # -----------------------------
        # KD-Tree
        # -----------------------------

        start = time.perf_counter()

        kd_results = self.kd_tree.search(
            query,
            k
        )

        kd_time = (
            time.perf_counter() - start
        ) * 1000

        benchmark_results["kd_tree"] = {
            "time_ms": kd_time,
            "results": [
                {
                    "id": item.id,
                    "distance": distance
                }
                for item, distance in kd_results
            ]
        }

        # -----------------------------
        # HNSW
        # -----------------------------

        start = time.perf_counter()

        hnsw_results = self.hnsw.search(
            query,
            k=k,
            ef=max(50, k)
        )

        hnsw_time = (
            time.perf_counter() - start
        ) * 1000

        benchmark_results["hnsw"] = {
            "time_ms": hnsw_time,
            "results": [
                {
                    "id": item.id,
                    "distance": distance
                }
                for item, distance in hnsw_results
            ]
        }

        return benchmark_results

    # --------------------------------------------------
    # HNSW information
    # --------------------------------------------------

    def hnsw_info(self) -> dict:

        with self.lock:
            return self.hnsw.get_info()

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    def stats(self) -> dict:

        with self.lock:
            return {
                "dimensions": self.dimensions,
                "metric": self.metric,
                "count": len(self.items),
                "brute_force_size": self.brute_force.size(),
                "kd_tree_size": self.kd_tree.size(),
                "hnsw_size": self.hnsw.size()
            }