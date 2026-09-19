from vector_core import VectorItem, get_distance_function


class KDNode:
    def __init__(
        self,
        item: VectorItem,
        axis: int
    ):
        self.item = item
        self.axis = axis
        self.left = None
        self.right = None


class KDTree:
    def __init__(self, dimensions: int, metric: str = "euclidean"):
        self.dimensions = dimensions
        self.root = None
        self.distance_function = get_distance_function(metric)

    def insert(self, item: VectorItem) -> None:
        self.root = self._insert(
            self.root,
            item,
            depth=0
        )

    def _insert(
        self,
        node: KDNode | None,
        item: VectorItem,
        depth: int
    ) -> KDNode:

        if node is None:
            axis = depth % self.dimensions
            return KDNode(item, axis)

        axis = node.axis

        if item.emb[axis] < node.item.emb[axis]:
            node.left = self._insert(
                node.left,
                item,
                depth + 1
            )
        else:
            node.right = self._insert(
                node.right,
                item,
                depth + 1
            )

        return node

    def search(
        self,
        query: list[float],
        k: int = 5
    ) -> list[tuple[VectorItem, float]]:

        results = []

        self._search(
            self.root,
            query,
            k,
            results
        )

        results.sort(key=lambda x: x[1])

        return results[:k]

    def _search(
        self,
        node: KDNode | None,
        query: list[float],
        k: int,
        results: list
    ) -> None:

        if node is None:
            return

        distance = self.distance_function(
            query,
            node.item.emb
        )

        results.append(
            (node.item, distance)
        )

        axis = node.axis

        if query[axis] < node.item.emb[axis]:
            first = node.left
            second = node.right
        else:
            first = node.right
            second = node.left

        self._search(
            first,
            query,
            k,
            results
        )

        self._search(
            second,
            query,
            k,
            results
        )

    def size(self) -> int:
        return self._count(self.root)

    def _count(self, node: KDNode | None) -> int:
        if node is None:
            return 0

        return (
            1
            + self._count(node.left)
            + self._count(node.right)
        )