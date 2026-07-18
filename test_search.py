import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="ticket_kb", metadata={"hnsw:space": "cosine"})

query = "package never arrived even though it says delivered"
query_embedding = model.encode(query).tolist()
results = collection.query(query_embeddings=[query_embedding], n_results=3)

for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"{doc} | distance={dist:.3f} | similarity={1-dist:.3f}")