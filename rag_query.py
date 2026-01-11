import os
from dotenv import load_dotenv

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# =========================
# Configuration
# =========================
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "hotel_docs"

SYSTEM_PROMPT = (
    "You are a hotel concierge assistant.\n"
    "You MUST answer using only the provided context.\n"
    "If the answer is not in the context, do NOT guess.\n"
    "Instead say:\n"
    "\"I don’t have that information in the hotel documents. "
    "For the most accurate details, please contact the hotel directly at +1 305 993 3300.\"\n"
    "Be concise, professional, and warm.\n"
)

# =========================
# Load Index
# =========================
def load_index():
    # Explicitly load .env (more reliable)
    load_dotenv(dotenv_path=".env")

    # Debug check (safe to keep for now)
    print("DEBUG: OPENAI_API_KEY loaded?", bool(os.getenv("OPENAI_API_KEY")))

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Check your .env file.")

    # Models
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        system_prompt=SYSTEM_PROMPT,
        temperature=0,
    )

    # Connect to existing Chroma DB (NO re-embedding)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    return VectorStoreIndex.from_vector_store(vector_store)

# =========================
# Main Chat Loop
# =========================
def main():
    if not os.path.isdir(PERSIST_DIR):
        raise FileNotFoundError("Missing chroma_db. Run: python ingest.py")

    index = load_index()

    query_engine = index.as_query_engine(
        similarity_top_k=4,
        response_mode="compact",
    )

    print("\nHotel GenAI Concierge (RAG). Type 'exit' to quit.\n")

    while True:
        q = input("Q: ").strip()

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        # Prevent accidental multi-question pastes
        q = q.splitlines()[0].strip()

        response = query_engine.query(q)

        # Robust response extraction (fixes Empty Response issue)
        answer = ""
        if hasattr(response, "response") and isinstance(response.response, str):
            answer = response.response.strip()

        if not answer:
            answer = str(response).strip()

        if not answer or answer.lower() == "empty response":
            answer = (
                "I don’t have that information in the hotel documents. "
                "For the most accurate details, please contact the hotel directly at +1 305 993 3300."
            )

        print("\nA:", answer)

        # Show sources for grounding / demo credibility
        if getattr(response, "source_nodes", None):
            print("\nSources:")
            for i, sn in enumerate(response.source_nodes, start=1):
                meta = sn.node.metadata or {}
                fname = meta.get("file_name") or meta.get("filename") or "unknown"
                snippet = sn.node.get_text().replace("\n", " ").strip()[:220]
                print(f"  {i}. {fname} — {snippet}...")

        print("\n" + "-" * 60 + "\n")

# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    main()
