# ChatBot Project

This project is a **ChatBot** built using Azure Functions and FAISS (Facebook AI Similarity Search) to respond to user queries based on trained embeddings data. The chatbot processes user queries, retrieves relevant information from the embeddings, and generates appropriate responses.

---

## Features
- **Query Processing**: Accepts user queries and processes them using trained embeddings.
- **Response Generation**: Generates responses based on the most relevant information retrieved from the embeddings.
- **Clarification Handling**: If the query is ambiguous, the chatbot asks clarifying questions to provide a more accurate response.
- **Error Handling**: Handles invalid inputs and unexpected errors gracefully.

---

## Prerequisites
Before running the project, ensure you have the following installed:
1. **Python 3.8 or higher**
2. **Azure Functions Core Tools** (for local development)
3. **FAISS** (for similarity search)
4. **Azure SDK for Python** (`azure-functions`)

---
