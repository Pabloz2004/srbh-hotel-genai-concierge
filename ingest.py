import os
from dotenv import load_dotenv

# Load OpenAI API key from .env
load_dotenv(dotenv_path=".env")

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "hotel_docs"


def main():
    # Safety checks
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"'{DATA_DIR}' folder not found. Create it and add .txt files."
        )

    # Load documents
    docs = SimpleDirectoryReader(DATA_DIR).load_data()
    if not docs:
        raise ValueError("No documents found in the data folder.")

    # Use OpenAI embeddings
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # Persistent Chroma DB
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # OPTIONAL: Clear collection each run to avoid duplicates while testing
    # Comment these lines out if you want incremental adds
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # ✅ IMPORTANT: Use StorageContext so vectors are actually written to Chroma
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build the index (writes embeddings into Chroma)
    VectorStoreIndex.from_documents(docs, storage_context=storage_context)

    print(f"✅ Indexed {len(docs)} documents into '{PERSIST_DIR}'")
    print(f"✅ Chroma collection '{COLLECTION_NAME}' vector count: {collection.count()}")


if __name__ == "__main__":
    main()
