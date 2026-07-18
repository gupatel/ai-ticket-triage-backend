import json
import chromadb
from fastembed import TextEmbedding

# Load the embedding model (runs locally, free, no API key needed)
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Set up a persistent ChromaDB client (saves to disk, not just memory)
client = chromadb.PersistentClient(path="./chroma_db")

# Create (or get) a collection - think of this like a table
collection = client.get_or_create_collection(
    name="ticket_kb",
    metadata={"hnsw:space": "cosine"}
)

# Load the knowledge base
with open("data/tickets_kb.json", "r") as f:
    tickets = json.load(f)

# Embed each ticket's "issue" text and store it
for ticket in tickets:
    embedding = list(model.embed([ticket["issue"]]))[0].tolist()

    collection.add(
        ids=[ticket["id"]],
        embeddings=[embedding],
        documents=[ticket["issue"]],
        metadatas=[{
            "category": ticket["category"],
            "priority": ticket["priority"],
            "resolution": ticket["resolution"]
        }]
    )

print(f"Successfully embedded {len(tickets)} tickets into ChromaDB.")