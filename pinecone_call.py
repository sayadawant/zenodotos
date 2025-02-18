import os
import pinecone
import openai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

load_dotenv()

# Retrieve API keys from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") 


client = pinecone.Pinecone(api_key=PINECONE_API_KEY)

index = client.Index(INDEX_NAME)

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text: str) -> list:
    """Get embedding vector from OpenAI for the provided text."""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-large",
            dimensions=1024
        )
        return response.data[0].embedding  
    except Exception as e:
        print(f"Error fetching embedding: {e}")
        return None

def query_pinecone(top_k=5):
    """Prompt user for query text and retrieve relevant documents."""
    query_text = input("\nEnter your query: ")  # Ask user for input    

    query_embedding = get_embedding(query_text)
    if query_embedding is None:
        print("Failed to get query embedding.")
        return
    
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    print("\n**Top Matches in Vector DB:**")
    for match in results["matches"]:
        metadata = match["metadata"]
        text = metadata.get("text", "No text available")
        source = metadata.get("source", "Unknown Source")  # Get "source" field
        doc_id = match.get("id", "No ID")  # Get ID from the response
        
        print(f"ID: {doc_id} | Score: {match['score']} | Source: {source}")
        print(f"   ➜ {text}\n")





# Example Query
if __name__ == "__main__":
    query_pinecone()
