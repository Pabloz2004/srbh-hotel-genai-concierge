# GenAI Hotel Concierge (RAG Prototype)

## Overview

I have built this project for my personal education it's a GenAI prototype to explore Retrieval-Augmented Generation (RAG).

I have modeled this assistant after The St. Regis Bal Harbour where I work, but it is not affiliated with or endorsed by any hotel brand. All data used is public, simulated, or fictionalized and intended for educational and portfolio purposes only.

The goal was to build a concierge assistant that answers only from verified documents and knowledge and safely escalates when information is missing.

---

## Project Structure
hotel-genai-concierge/
├── data/
│   ├── amenities.txt
│   ├── dining.txt
│   ├── experiences.txt
│   ├── local_guide.txt
│   ├── policies.txt
│   └── rooms_and_suites.txt
├── ingest.py
├── rag_query.py
├── requirements.txt
├── .gitignore
└── readme.md


---

## How It Works

1. Ii ingest hotel imformation from the `data/` folder  
2. This documents are embedded using OpenAI embeddings  
3. The mbeddings are stored in a persistent Chroma vector database  
4. User questions retrieve the most relevant context  
5. The model answers *only from retrieved content*  
6. If the answer is unavailable, the assistant advises contacting the hotel official number directly  

---

## Key Features

- Retrieval-Augmented Generation (RAG)
- Hallucination-resistant answers
- Persistent local vector database
- Concierge-style responses
- Safe fallback for unknown questions

---

## Cost

This project is **very low cost**:
- Embeddings are generated once during ingestion
- Each question costs only around ~$0.00002
- Suitable for demos, learning, and small prototypes

---

## Disclaimer

This project is a **personal educational prototype** and is **not an official system** used by any hotel or hospitality brand.

---

## Author

Pablo López Franco
Computer Science Student | GenAI & Software Engineering

