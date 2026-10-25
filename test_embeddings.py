from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Sample document chunks
chunks = [
    "Python is a programming language.",
    "Machine learning is a branch of artificial intelligence.",
    "FAISS is used for efficient vector similarity search."
]

# Generate embeddings and create a vector store
vector_store = FAISS.from_texts(
    texts=chunks,
    embedding=embeddings
)

print("Vector store created successfully!")

# Search for the most relevant text
results = vector_store.similarity_search(
    "What is machine learning?",
    k=2
)

print("\nSearch results:")

for result in results:
    print(result.page_content)