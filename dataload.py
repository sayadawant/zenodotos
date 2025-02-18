import os
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Get keys and environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME").strip()

# Global constants for index configuration
EMBEDDING_DIM = 1024  # Adjust to match your chosen model's dimension

# Instantiate the OpenAI client using the new module-level instantiation
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Instantiate the Pinecone client using the new API
pc = Pinecone(api_key=PINECONE_API_KEY)

def create_or_connect_index(name: str, dimension: int, metric: str = "cosine"):
    """Create a new Pinecone index if it doesn't exist, else connect to it."""
    existing_indexes = pc.list_indexes().names()
    if name not in existing_indexes:
        logging.info(f"Creating Pinecone index '{name}'...")
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud='aws',         # Assuming AWS for free-tier accounts
                region=PINECONE_ENV  # Defined in your .env file (e.g., us-east1)
            )
        )
    else:
        logging.info(f"Pinecone index '{name}' exists. Connecting...")
    return pc.Index(name)

def get_embedding(text: str) -> list:
    """
    Retrieve the embedding for the provided text using the new OpenAI client.
    
    This function calls the new client method and accesses the embedding via attribute access.
    """
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-large",
            dimensions=1024,
        )
        # Use attribute access on the pydantic model response:
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error fetching embedding: {e}")
        raise

def load_csv_data(file_path: str) -> pd.DataFrame:
    """Load CSV file into a pandas DataFrame assuming semicolon delimiters."""
    try:
        df = pd.read_csv(file_path, delimiter=";")
        # Optionally normalize header names to lowercase to match our code
        df.columns = df.columns.str.lower()
        logging.info(f"Loaded {len(df)} records from {file_path}.")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV: {e}")
        raise


def clean_metadata(metadata: dict) -> dict:
    # Filter out keys with missing (NaN) values
    return {k: v for k, v in metadata.items() if pd.notna(v)}

def upsert_embeddings(index, df: pd.DataFrame, batch_size: int = 100):
    vectors = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Upserting records"):
        metadata = {
            "id": row["id"],
            "source": row["source"],
            "url": row["url"],
            "title": row["title"],
            "timestamp": row["timestamp"],
            "text": row["text"]
        }
        metadata = clean_metadata(metadata)
        embedding = get_embedding(row["text"])
        vectors.append((str(i), embedding, metadata))
        
        if len(vectors) >= batch_size:
            index.upsert(vectors=vectors)
            vectors = []
    
    if vectors:
        index.upsert(vectors=vectors)
    logging.info("✅ Data stored in Pinecone!")

def main(csv_file: str):
    index = create_or_connect_index(INDEX_NAME, EMBEDDING_DIM)
    df = load_csv_data(csv_file)
    upsert_embeddings(index, df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a CSV and store text embeddings in Pinecone.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file.")
    args = parser.parse_args()
    main(args.csv_file)
