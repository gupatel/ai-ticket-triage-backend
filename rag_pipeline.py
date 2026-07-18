import os
import json
from dotenv import load_dotenv
import chromadb
from fastembed import TextEmbedding
from groq import Groq

load_dotenv()

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="ticket_kb",
    metadata={"hnsw:space": "cosine"}
)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def retrieve_similar_tickets(query: str, n_results: int = 3):
    query_embedding = list(model.embed([query]))[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    similar = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        similar.append({
            "issue": doc,
            "category": meta["category"],
            "priority": meta["priority"],
            "resolution": meta["resolution"],
            "similarity_score": round(max(0, 1 - dist), 3)
        })
    return similar


def triage_ticket(new_ticket_text: str):
    similar_tickets = retrieve_similar_tickets(new_ticket_text)

    context = "\n\n".join([
        f"Past issue: {t['issue']}\nCategory: {t['category']}\nPriority: {t['priority']}\nResolution: {t['resolution']}"
        for t in similar_tickets
    ])

    prompt = f"""You are a support ticket triage assistant. Based on similar past tickets below, classify the new ticket and suggest a resolution.

Similar past tickets:
{context}

New ticket: {new_ticket_text}

Respond ONLY in this JSON format, nothing else:
{{
  "category": "...",
  "priority": "High|Medium|Low",
  "suggested_response": "..."
}}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    result = json.loads(raw)
    result["retrieved_context"] = similar_tickets
    return result


if __name__ == "__main__":
    test_ticket = "I was billed twice for my subscription this month"
    result = triage_ticket(test_ticket)
    print(json.dumps(result, indent=2))