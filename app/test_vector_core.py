from vector_core import (
    VectorItem,
    euclidean,
    cosine,
    manhattan,
    get_distance_function,
)


a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

print("Euclidean:", euclidean(a, b))
print("Cosine:", cosine(a, b))
print("Manhattan:", manhattan(a, b))

item = VectorItem(
    id=1,
    metadata="Test vector",
    category="test",
    emb=a,
)

print("VectorItem:", item)

print(
    "Selected metric:",
    get_distance_function("cosine").__name__
)