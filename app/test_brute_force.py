from vector_core import VectorItem
from brute_force import BruteForce


db = BruteForce(metric="euclidean")

db.insert(
    VectorItem(
        id=1,
        metadata="Apple",
        category="fruit",
        emb=[1.0, 1.0, 1.0],
    )
)

db.insert(
    VectorItem(
        id=2,
        metadata="Banana",
        category="fruit",
        emb=[2.0, 2.0, 2.0],
    )
)

db.insert(
    VectorItem(
        id=3,
        metadata="Car",
        category="vehicle",
        emb=[10.0, 10.0, 10.0],
    )
)

query = [1.1, 1.1, 1.1]

results = db.search(query, k=2)

print("Database size:", db.size())

print("\nTop results:")

for result in results:
    print(
        f"ID: {result.item.id}, "
        f"Metadata: {result.item.metadata}, "
        f"Distance: {result.distance}"
    )

print("\nRemoving ID 2:", db.remove(2))
print("Database size after removal:", db.size())