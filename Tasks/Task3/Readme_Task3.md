-----
**HogwartsQ&A**

Welcome to the **HogwartsQ&A** project! This repository contains a Retrieval-Augmented Generation (RAG) system that allows users to ask questions related to characters, spells, locations, and magical events from the book "Harry Potter and the Prisoner of Azkaban." The system retrieves contextually accurate information and generates lore-true responses, complete with quotes and references.
## Data Collection & Ingestion
1. Download the **"Harry Potter and the Prisoner of Azkaban"** PDF from the provided link.
1. Parse the PDF into a machine-readable format, preserving the chapters and significant sections. The text is structured to facilitate easy retrieval based on context.
## Data Chunking & Preprocessing
- The text is divided into smaller chunks of 100-150 words, ensuring that each chunk contains coherent, self-contained information. This chunking process helps maintain context during retrieval.
## Embedding Generation
- Each text chunk is converted into dense vector representations using the pre-trained embedding model **all-MiniLM-L6-v2** from Sentence Transformers. This step captures the semantic meaning of the text, making it easier to retrieve relevant chunks based on user queries.
## Vector Database Integration
- The embeddings are stored in a vector database (e.g., **ChromaDB**) for efficient similarity searches and quick lookups. This setup allows for fast retrieval of contextually relevant information.
## Query Handling & Retrieval
- A query pipeline processes user queries by converting them into embeddings. The pipeline retrieves the top N most relevant text chunks from the vector database based on the similarity to the query embedding.
## Contextual Response Generation
- Retrieved text chunks are fed into a generative language model (such as **Gemini** or **LLaMA**) to create coherent responses. The generated output incorporates quotes and references, maintaining the tone of the Harry Potter universe.
## API Usage
The RAG system is served via FastAPI. You can interact with the API using the /query endpoint. This endpoint accepts POST requests with a JSON body containing the user's question.

A Retrieval-Augmented Generation (RAG) chatbot combines information retrieval with natural language generation to provide contextually accurate and coherent responses.
