from dataclasses import dataclass

from vector_core import VectorItem, get_distance_function


@dataclass
class SearchResult:
    item: VectorItem
    distance: float


class BruteForce:
    def __init__(self, metric: str = "euclidean"):
        self.items: list[VectorItem] = []
        self.distance_function = get_distance_function(metric)

    def insert(self, item: VectorItem) -> None:
        self.items.append(item)

    def remove(self, item_id: int) -> bool:
        for i, item in enumerate(self.items):
            if item.id == item_id:
                self.items.pop(i)
                return True

        return False

    def search(
        self,
        query: list[float],
        k: int = 5
    ) -> list[SearchResult]:

        results = []

        for item in self.items:
            distance = self.distance_function(query, item.emb)
            results.append(SearchResult(item, distance))

        results.sort(key=lambda result: result.distance)

        return results[:k]

    def size(self) -> int:
        return len(self.items)