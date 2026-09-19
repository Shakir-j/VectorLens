from vector_core import VectorItem
from hnsw import HNSW


hnsw = HNSW(metric="euclidean")


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
        metadata="Orange",
        category="fruit",
        emb=[3.0, 3.0, 3.0]
    ),
    VectorItem(
        id=4,
        metadata="Car",
        category="vehicle",
        emb=[10.0, 10.0, 10.0]
    ),
    VectorItem(
        id=5,
        metadata="Bike",
        category="vehicle",
        emb=[8.0, 8.0, 8.0]
    ),
]


# Insert vectors
for item in items:
    hnsw.insert(item)


print("HNSW size:", hnsw.size())
print("Entry point:", hnsw.entry_point)
print("Top layer:", hnsw.top_layer)


# Display graph structure
print("\nGraph:")

for node_id, node in hnsw.graph.items():
    print(
        f"ID {node_id}: "
        f"level={node.max_level}, "
        f"neighbors={node.neighbors}"
    )


# Perform a nearest-neighbor search
query = [1.1, 1.1, 1.1]

results = hnsw.search(
    query,
    k=3,
    ef=50
)


print("\nSearch results:")

for item, distance in results:
    print(
        f"ID: {item.id}, "
        f"Metadata: {item.metadata}, "
        f"Category: {item.category}, "
        f"Distance: {distance}"
    )