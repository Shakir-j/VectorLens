from vector_core import VectorItem
from kd_tree import KDTree


tree = KDTree(
    dimensions=3,
    metric="euclidean"
)

items = [
    VectorItem(
        id=1,
        metadata="Apple",
        category="fruit",
        emb=[1.0, 1.0, 1.0]
    ),
    VectorItem(
        id=2,
        metadata="Banana",
        category="fruit",
        emb=[2.0, 2.0, 2.0]
    ),
    VectorItem(
        id=3,
        metadata="Car",
        category="vehicle",
        emb=[10.0, 10.0, 10.0]
    ),
]

for item in items:
    tree.insert(item)

query = [1.1, 1.1, 1.1]

results = tree.search(
    query,
    k=2
)

print("Tree size:", tree.size())

print("\nTop results:")

for item, distance in results:
    print(
        f"ID: {item.id}, "
        f"Metadata: {item.metadata}, "
        f"Distance: {distance}"
    )