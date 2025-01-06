import faiss
import json
import numpy as np
from openai import OpenAI
from pymongo import MongoClient
import uuid
from datetime import datetime
from sklearn.preprocessing import normalize 
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from fuzzywuzzy import process, fuzz

# Set up OpenAI API Key
OPENAI_API_KEY = "API KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

STORAGE_ACCOUNT_URL = "https://storagehwaimipi.blob.core.windows.net"

# Authenticate using Managed Identity
credential = DefaultAzureCredential()
blob_service_client = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=credential)

# MongoDB configuration
MONGO_URI = "URI"
DB_NAME = "chatbot"
COLLECTION_NAME = "chat_history"

# Establish MongoDB connection
def get_mongo_client():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("Connected to MongoDB.")
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

# Initialize MongoDB client and database
mongo_client = get_mongo_client()
db = mongo_client[DB_NAME] if mongo_client is not None else None
collection = db[COLLECTION_NAME] if db is not None else None


def load_faiss_index():
    try:
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=credential)
        blob_client = blob_service_client.get_blob_client(container="hwai-monthly-refresh", blob="ChatBot/Embeddings_Data/PC/PC_index_userGuide.index")
        index_data = blob_client.download_blob().readall()

        index_path = "/tmp/faiss_index.index"
        with open(index_path, "wb") as f:
            f.write(index_data)

        index = faiss.read_index(index_path)
        print("FAISS index loaded successfully.")
        return index
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

# Load metadata from Azure Blob Storage
def load_metadata():
    try:
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=credential)
        blob_client = blob_service_client.get_blob_client(container="hwai-monthly-refresh", blob="ChatBot/Embeddings_Data/PC/PC_index_userGuide_metadata.json")
        metadata_data = blob_client.download_blob().readall()

        metadata = json.loads(metadata_data)
        print("Metadata loaded successfully.")
        return metadata
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


# Generate embedding for a query
def get_embedding(query, model="text-embedding-ada-002"):
    query = query.replace("\n", " ")
    response = client.embeddings.create(input=[query], model=model)
    embedding = response.data[0].embedding
    return normalize([embedding])[0]  # Normalize embedding for cosine similarity

# CHANGE START: Metadata filtering
def filter_metadata(metadata, filters):
    """
    Filter metadata based on specific attributes.
    """
    filtered_results = []
    for item in metadata:
        # Apply document-based filters
        if "chapter" in filters and item.get("chapter") != filters["chapter"]:
            continue
        if "tags" in filters and not any(tag in item.get("tags", []) for tag in filters["tags"]):
            continue
        # Apply FAQ-based filters
        if "category" in filters and item.get("category") != filters["category"]:
            continue
        if "query_keywords" in filters:
            query = item.get("query", "").lower()
            if not any(keyword.lower() in query for keyword in filters["query_keywords"]):
                continue
        filtered_results.append(item)
    return filtered_results

# Fuzzy matching for FAQs
def fuzzy_match_faq(metadata, query, threshold=80):
    """
    Perform fuzzy matching on FAQ queries and return results above the threshold.
    """
    faq_queries = [item.get("query", "") for item in metadata]
    matches = process.extract(query, faq_queries, scorer=fuzz.partial_ratio, limit=5)
    return [
        metadata[i] for i, (_, score) in enumerate(matches) if score >= threshold
    ]

# CHANGE START: Re-ranking with Cross-Encoder
def re_rank_results(query, results):
    """
    Re-rank results using GPT-4 cross-encoder for contextual relevance.
    """
    re_ranked = []
    for result in results:
        prompt = f"""
        Query: {query}
        Document: {result['metadata']}
        Rate the relevance of the document to the query on a scale of 0 to 1.
        """
        relevance_score = float(generate_response(prompt))  # Use GPT-4 for scoring
        re_ranked.append((result, relevance_score))
    
    # Sort results by relevance score in descending order
    return sorted(re_ranked, key=lambda x: x[1], reverse=True)

def hybrid_query_retrieval(query, query_embedding, index, metadata, filters=None, top_k=3):
    """
    Combine metadata filtering, fuzzy matching, and FAISS retrieval.
    """
    filtered_metadata = filter_metadata(metadata, filters) if filters else metadata
    fuzzy_results = fuzzy_match_faq(filtered_metadata, query)
    
    # Perform FAISS search on filtered metadata
    filtered_indices = [i for i, item in enumerate(metadata) if item in filtered_metadata]
    distances, indices = index.search(np.array([query_embedding]).astype("float32"), top_k)

    # Combine results
    hybrid_results = []
    for rank, idx in enumerate(indices[0]):
        if idx in filtered_indices:
            hybrid_results.append({
                "rank": rank + 1,
                "distance": distances[0][rank],
                "metadata": metadata[idx]
            })
    return hybrid_results + fuzzy_results[:top_k]

# Perform similarity search using cosine similarity
def search_faiss_index(query_embedding, index, metadata, top_k=3):
    query_vector = np.array(query_embedding).reshape(1, -1).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            results.append({
                "rank": i + 1,
                "distance": distances[0][i],
                "metadata": metadata[idx]
            })
        else:
            print(f"Warning: Index {idx} out of metadata bounds.")
    return results

# Construct a cleaner prompt
def construct_prompt(query, clarifications, history, relevant_docs):
    history_snippet = "\n".join(
        [f"User: {item['query']}\nAssistant: {item['response']}" for item in history]
    ) if history else "No prior interaction."

    clarifications_snippet = "\n".join(clarifications) if clarifications else "No clarifications provided."

    relevant_context = "\n".join([
        f"Section: {doc['metadata'].get('section', 'Unknown Section')}\nContent: {doc['metadata'].get('content', 'No Content')}"
        for doc in relevant_docs
    ]) if relevant_docs else "No relevant documents found."

    prompt = f"""
    You are the HealthworksAI ChatBot, assisting users with plan comparison and related queries.

    Conversation History:
    {history_snippet}

    Clarifications:
    {clarifications_snippet}

    Relevant Context:
    {relevant_context}

    User Query:
    {query}

    Provide a clear, helpful, and concise response.
    """
    return prompt.strip()

# Debugging the results
def debug_results(fetched_results):
    print("\nDebugging Results:")
    for result in fetched_results:
        print(f"Rank: {result['rank']}, Distance: {result['distance']}")
        print(f"Metadata: {json.dumps(result['metadata'], indent=2)}\n")

# Retrieve relevant history
def get_relevant_history(session_id, max_items=3):
    """
    Retrieve the last 'max_items' queries and responses for a given session ID from MongoDB.
    """
    if collection is None:
        print("MongoDB collection is not available.")
        return []

    try:
        history = list(collection.find({"session_id": session_id}))
        history.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
        return [{"query": h["query"], "response": h["response"]} for h in history[:max_items]]
    except Exception as e:
        print(f"Error retrieving history: {e}")
        return []


def save_chat_history(session_id, user_id, client_id, dashboard_id, query, response, clarifications=None):
    """
    Save the current query, response, and any clarifications to MongoDB, along with user, client, and dashboard details.
    """
    if collection is None:
        print("MongoDB collection is not available. Unable to save chat history.")
        return

    try:
        # Prepare chat history object
        chat_history = {
            "session_id": session_id,
            "user_id": user_id,
            "client_id": client_id,
            "dashboard_id": dashboard_id,
            "timestamp": datetime.utcnow(),
            "query": query,
            "response": response,
            "clarifications": clarifications or []  # Save clarifications if provided
        }

        # Insert the document into MongoDB
        collection.insert_one(chat_history)
        print("Chat history saved successfully.")
    except Exception as e:
        print(f"Error saving chat history: {e}")


# Query Classification
def classify_query(query):
    """
    Classify the query as Ambiguous, Simple, or Complex.
    """
    prompt = f"""
    You are a query classifier for the HealthworksAI ChatBot. 
    Your task is to classify the user query into one of the following categories:
    - **Ambiguous**: The query has unclear intent, lacks specificity, or could have multiple interpretations.
    - **Simple**: The query is straightforward and can be answered directly using available information.
    - **Complex**: The query requires extensive analysis or human intervention.


    Consider the following factors:
    1. Analyze the user's query and identify any unclear terms or missing information that may require clarification.
    2. If the query involves nuanced decision-making, highly technical terms, or complex logic, classify it as Complex.

    Examples of Ambiguous Queries:
    - "How can I sort?" (Unclear what needs to be sorted or on what basis.)
    - "Tell me about plan benefits." (Too broad; does not specify which plan or benefit type.)
    
    Examples of Simple Queries:
    - "Do we show enrollment growth in Plan Comparison?"

    Examples of Complex Queries:
    - "Compare the total value of plans across multiple counties considering crosswalk adjustments for the last three years."
    - "How do crosswalked benefits differ across plans when considering OOPC for Part D drug costs?"

    User Query: "{query}"

    Respond with only one word: Ambiguous, Simple, or Complex.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for query classification."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content.strip().lower()

# Generate response
def generate_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


# Generate clarifying question
def generate_clarifying_question(query, context_snippets):
    """
    Generate a clarifying question based on the user's ambiguous query and context.
    """
    messages = [
        {"role": "system", "content": "You are the HealthworksAI ChatBot. Generate a clarifying question to refine the query."},
        {"role": "user", "content": f"Query: {query}\nContext: {context_snippets}"}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=50
    )
    return response.choices[0].message.content.strip()

def query_and_generate_response(query, session_id=None, user_id=None, client_id=None, dashboard_id=None, clarifications=None, top_k=3):
    session_id = session_id or str(uuid.uuid4())
    index = load_faiss_index()
    metadata = load_metadata()

    if not index or not metadata:
        print("Failed to load FAISS index or metadata. Cannot proceed.")
        return {"status": "error", "message": "Failed to load FAISS index or metadata."}

    query_embedding = get_embedding(query)
    filters = {"chapter": "9. Download Plans", "tags": ["Download Plans", "Export Data"]}
    hybrid_results = hybrid_query_retrieval(query, query_embedding, index, metadata, filters, top_k)
    re_ranked_results = re_rank_results(query, hybrid_results)
    top_results = [item[0] for item in re_ranked_results[:top_k]]
    
    fetched_results = search_faiss_index(query_embedding, index, metadata, top_k)
    debug_results(fetched_results)

    # Retrieve session history and clarifications
    history = get_relevant_history(session_id)
    previous_clarification = next((item for item in history if "clarifying_question" in item), None)

    # Classify query
    classification = classify_query(query)
    print(f"Query has been classified as {classification}")

    if classification == "ambiguous":
        if previous_clarification and clarifications:
            # Use previous clarifications to construct a response
            prompt = construct_prompt(query, clarifications, history, fetched_results)
            response = generate_response(prompt)

            # Save chat history
            save_chat_history(session_id, user_id, client_id, dashboard_id, query, response, clarifications)

            return {
                "status": "success",
                "user_id": user_id,
                "session_id": session_id,
                "query": query,
                "response": response,
                "clarifications": clarifications
            }
        else:
            # Generate a new clarifying question
            context_snippets = "\n".join([str(doc["metadata"]) for doc in fetched_results])
            clarifying_question = generate_clarifying_question(query, context_snippets)

            # Save clarification prompt to MongoDB
            save_chat_history(session_id, user_id, client_id, dashboard_id, query, None, clarifications)

            return {
                "status": "needs_clarification",
                "clarifying_question": clarifying_question,
                "context_snippets": context_snippets,
                "clarifications": clarifications or []  # Maintain previous clarifications
            }

    # Construct final prompt using clarifications
    prompt = construct_prompt(query, clarifications, history, fetched_results)
    response = generate_response(prompt)

    # Save chat history
    save_chat_history(session_id, user_id, client_id, dashboard_id, query, response, clarifications)

    return {
        "status": "success",
        "user_id": user_id,
        "session_id": session_id,
        "query": query,
        "response": response,
        "clarifications": clarifications or []
    }
