import os
import logging
import pinecone
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Retrieve configuration from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME").strip()
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Desired embedding configuration for "text-embedding-3-large"
EMBEDDING_DIM = 1024  # (Note: if the model returns a different dimension, update accordingly)
EMBEDDING_MODEL = "text-embedding-3-large"

# Instantiate the OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone using the new API and include the environment parameter
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
existing_indexes = pc.list_indexes().names()
logging.info(f"Existing indexes: {existing_indexes}")
if INDEX_NAME not in existing_indexes:
    logging.error(f"Index '{INDEX_NAME}' does not exist. Please load your data into Pinecone first.")
    exit(1)
else:
    logging.info(f"Connecting to existing index '{INDEX_NAME}'...")
    index = pc.Index(INDEX_NAME)

def get_embedding(text: str) -> list:
    """Get embedding vector from OpenAI using the 'text-embedding-3-large' model."""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL,
            dimensions=1024
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error fetching embedding: {e}")
        return None

def query_pinecone_results(query_text: str, top_k: int = 10) -> list:
    """Query Pinecone with the provided query text and return top_k results."""
    embedding = get_embedding(query_text)
    if embedding is None:
        raise ValueError("Failed to obtain query embedding.")
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return results.get("matches", [])

def filter_blog_entries(results: list, blog_max: int = 2) -> list:
    """
    Limit the number of results with type equal to 'blog' to at most blog_max.
    Assumes the CSV column 'type' was mapped to the metadata field 'source'.
    """
    blog_entries = [r for r in results if r["metadata"].get("source", "").lower() == "blog"]
    non_blog_entries = [r for r in results if r["metadata"].get("source", "").lower() != "blog"]

    # Sort blog entries by score (descending) and keep only the top blog_max ones.
    blog_entries = sorted(blog_entries, key=lambda r: r["score"], reverse=True)[:blog_max]
    filtered = sorted(blog_entries + non_blog_entries, key=lambda r: r["score"], reverse=True)
    return filtered

def concatenate_texts(results: list) -> str:
    """Concatenate the 'text' fields from a list of Pinecone result matches."""
    texts = [r["metadata"].get("text", "") for r in results]
    return "\n".join(texts)

def get_summary(context: str, max_tokens: int = 2000) -> str:
    """
    Use GPT-4 to generate a summary of the provided context.
    The summary is limited to max_tokens.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful summarizer."},
                {"role": "user", "content": f"Summarize the following text concisely (max {max_tokens} tokens):\n\n{context}"}
            ],
            max_tokens=max_tokens
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        logging.error(f"Error fetching summary: {e}")
        return ""

def get_final_answer(summary: str, top_quotes: list, user_question: str) -> str:
    """
    Use GPT-4 to generate a final answer using the summary and direct quotes.
    The system prompt is stored in the environment variable SYSTEM_PROMPT.
    """
    quotes_text = "\n".join(
        [f"- {q['metadata'].get('text', 'No text available')}" for q in top_quotes]
    )
    
    prompt = (
        f"Based on the following summary of relevant documents:\n\n{summary}\n\n"
        f"And here are direct quotes from the top results:\n{quotes_text}\n\n"
        f"Answer the following question clearly and concisely:\n{user_question}"
    )
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        final_answer = response.choices[0].message.content.strip()
        return final_answer
    except Exception as e:
        logging.error(f"Error fetching final answer: {e}")
        return "Error generating final answer."

def main():
    user_question = input("Enter your question: ")
    logging.info("Querying Pinecone for relevant documents...")
    results = query_pinecone_results(user_question, top_k=10)
    filtered_results = filter_blog_entries(results, blog_max=2)
    top_quotes = sorted(filtered_results, key=lambda r: r["score"], reverse=True)[:2]
    concatenated_text = concatenate_texts(filtered_results)
    logging.info("Generating summary of the retrieved context...")
    summary = get_summary(concatenated_text, max_tokens=2000)
    logging.info("Generating final answer...")
    final_answer = get_final_answer(summary, top_quotes, user_question)
    print("\n=== Final Answer ===")
    print(final_answer)

if __name__ == "__main__":
    main()
