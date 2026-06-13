"""
Mini Vector Search Engine From Scratch

This project demonstrates the basic internal idea behind a vector database:
1. Convert text into vectors
2. Store document vectors
3. Convert a query into a vector
4. Compare vectors using cosine similarity
5. Return the most similar documents

No external vector database or embedding library is used.
This is for learning the fundamentals behind semantic/vector search.
"""

import math
from collections import Counter


documents = [
    {
        "id": 1,
        "text": "Python is used for machine learning and data science",
        "metadata": {"category": "technology"},
    },
    {
        "id": 2,
        "text": "Hospitals use doctors and nurses to treat patients",
        "metadata": {"category": "health"},
    },
    {
        "id": 3,
        "text": "Cars need fuel engines and wheels",
        "metadata": {"category": "automobile"},
    },
    {
        "id": 4,
        "text": "Machine learning helps computers learn from data",
        "metadata": {"category": "technology"},
    },
    {
        "id": 5,
        "text": "Doctors prescribe medicine for fever and pain",
        "metadata": {"category": "health"},
    },
]


def tokenize(text):
    """
    Convert text into lowercase words.

    Example:
    "Machine Learning" -> ["machine", "learning"]
    """
    return text.lower().split()


def build_vocabulary(docs):
    """
    Build a unique sorted vocabulary from all documents.
    Each word in this vocabulary represents one dimension in the vector.
    """
    all_words = []

    for doc in docs:
        words = tokenize(doc["text"])
        all_words.extend(words)

    return sorted(set(all_words))


def text_to_vector(text, vocabulary):
    """
    Convert text into a word-count vector.

    Example:
    vocabulary = ["data", "doctor", "machine"]
    text = "machine data data"

    vector = [2, 0, 1]
    """
    words = tokenize(text)
    word_counts = Counter(words)

    vector = []

    for word in vocabulary:
        vector.append(word_counts[word])

    return vector


def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.

    Cosine similarity measures how close two vectors are in direction.
    Score range:
    - 1.0 means very similar
    - 0.0 means not similar
    """

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


def create_vector_store(docs, vocabulary):
    """
    Add vector representation to every document.
    This simulates storing embeddings in a vector database.
    """
    vector_store = []

    for doc in docs:
        vector_store.append(
            {
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
                "vector": text_to_vector(doc["text"], vocabulary),
            }
        )

    return vector_store


def search(query, vector_store, vocabulary, top_k=3, category_filter=None):
    """
    Search for the most similar documents.

    Parameters:
    query: User search query
    vector_store: Stored document vectors
    vocabulary: Vocabulary used for vector creation
    top_k: Number of results to return
    category_filter: Optional metadata filter
    """

    query_vector = text_to_vector(query, vocabulary)
    results = []

    for doc in vector_store:
        if category_filter is not None:
            if doc["metadata"]["category"] != category_filter:
                continue

        score = cosine_similarity(query_vector, doc["vector"])

        results.append(
            {
                "id": doc["id"],
                "text": doc["text"],
                "category": doc["metadata"]["category"],
                "score": round(score, 4),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":
    vocabulary = build_vocabulary(documents)
    vector_store = create_vector_store(documents, vocabulary)

    print("Vocabulary:")
    print(vocabulary)

    print("\nSearch Example 1:")
    query = "machine learning data"
    results = search(query, vector_store, vocabulary, top_k=3)

    for result in results:
        print(result)

    print("\nSearch Example 2 with Metadata Filter:")
    query = "fever medicine"
    results = search(
        query,
        vector_store,
        vocabulary,
        top_k=3,
        category_filter="health",
    )

    for result in results:
        print(result)