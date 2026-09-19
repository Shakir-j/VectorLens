from vector_core import VectorItem
from brute_force import BruteForce
from hnsw import HNSW


# --------------------------------------------------
# Create test vectors
# --------------------------------------------------

items = [
    VectorItem(1, "Apple", "fruit", [1.0, 1.0, 1.0]),
    VectorItem(2, "Banana", "fruit", [2.0, 2.0, 2.0]),
    VectorItem(3, "Orange", "fruit", [3.0, 3.0, 3.0]),
    VectorItem(4, "Mango", "fruit", [4.0, 4.0, 4.0]),
    VectorItem(5, "Car", "vehicle", [10.0, 10.0, 10.0]),
    VectorItem(6, "Bike", "vehicle", [8.0, 8.0, 8.0]),
    VectorItem(7, "Bus", "vehicle", [12.0, 12.0, 12.0]),
    VectorItem(8, "Train", "vehicle", [15.0, 15.0, 15.0]),
]


# --------------------------------------------------
# Create both indexes
# --------------------------------------------------

brute_force = BruteForce(metric="euclidean")
hnsw = HNSW(metric="euclidean")


# --------------------------------------------------
# Insert the same vectors into both
# --------------------------------------------------

for item in items:
    brute_force.insert(item)
    hnsw.insert(item)


# --------------------------------------------------
# Query
# --------------------------------------------------

query = [1.1, 1.1, 1.1]
k = 5


# --------------------------------------------------
# Brute Force search
# --------------------------------------------------

brute_results = brute_force.search(
    query,
    k=k
)


# --------------------------------------------------
# HNSW search
# --------------------------------------------------

hnsw_results = hnsw.search(
    query,
    k=k,
    ef=50
)


# --------------------------------------------------
# Display Brute Force results
# --------------------------------------------------

print("\nBrute Force Results:")

for result in brute_results:
    print(
        f"ID: {result.item.id}, "
        f"Metadata: {result.item.metadata}, "
        f"Distance: {result.distance}"
    )


# --------------------------------------------------
# Display HNSW results
# --------------------------------------------------

print("\nHNSW Results:")

for item, distance in hnsw_results:
    print(
        f"ID: {item.id}, "
        f"Metadata: {item.metadata}, "
        f"Distance: {distance}"
    )


# --------------------------------------------------
# Compare IDs
# --------------------------------------------------

brute_ids = [
    result.item.id
    for result in brute_results
]

hnsw_ids = [
    item.id
    for item, distance in hnsw_results
]


print("\nComparison:")
print("Brute Force IDs:", brute_ids)
print("HNSW IDs:       ", hnsw_ids)


if brute_ids == hnsw_ids:
    print("\nPASS: HNSW matches Brute Force.")
else:
    print("\nWARNING: HNSW results differ from Brute Force.")


# --------------------------------------------------
# Basic size check
# --------------------------------------------------

print("\nIndex sizes:")
print("Brute Force:", brute_force.size())
print("HNSW:", hnsw.size())