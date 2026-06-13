# Vector Search From Scratch

This project is a beginner-friendly implementation of a simple vector search engine using pure Python.

The goal of this project is to understand the core idea behind vector databases before using tools such as Pinecone, FAISS, Chroma, Milvus, Qdrant, Weaviate, or pgvector.

## What This Project Demonstrates

This project shows how vector search works internally at a basic level:

1. Text is converted into tokens.
2. A vocabulary is built from all documents.
3. Each document is converted into a numerical vector.
4. A user query is converted into a vector using the same vocabulary.
5. Cosine similarity is used to compare the query vector with document vectors.
6. The most similar documents are returned.
7. Metadata filtering is used to narrow search results.

## Why I Built This

Modern AI applications often use vector databases for semantic search and Retrieval-Augmented Generation, also known as RAG.

Before using a production vector database, I wanted to understand the basic mechanism behind:

- embeddings
- vectors
- similarity search
- cosine similarity
- metadata filtering
- top-k retrieval

This project is intentionally simple and does not use any external AI or vector database library.

## How It Works

A document such as:

# Clone and run
https://github.com/noim015/vector-search-from-scratch.git
cd vector-search-from-scratch
python main.py