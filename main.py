import os

import requests
from sklearn.metrics.pairwise import cosine_similarity

import chromadb
from google import genai
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


API_KEY = os.environ.get("API_KEY")
HUGGING_FACE_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")


class GenUtility:
    """
    Calls Gemini API for generating content (model : gemini-2.0-flash)
    """

    def __init__(self):
        self.model_client = genai.Client(api_key=API_KEY)

    def generate_response(self, query: str, context: str):
        prompt = f"""
        You are an expert AI assistant. Use the following context to provide a **detailed, accurate, and well-structured response** to the user query.
        ### Context:
        {context}
        ### User Query:
        {query}
        ### Instructions:
        - Utilize the provided context as a **primary knowledge base** to answer the query.
        - If the context lacks sufficient details, provide a **general explanation** while staying relevant.
        - Structure your response clearly with **key points, explanations, and examples if necessary**.
        - Ensure factual accuracy and avoid unnecessary assumptions.
        """
        response = self.model_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        print("===== GENERATED RESPONSE =====")
        print(f"{response.text}\n")
        return response.text


class EmbeddingUtility:
    """
    Token embedding utility.
    Calls hugging face API for embedding tokens.
    [Heavylifting done on upstream server :) ]
    """

    def __init__(self):
        self.embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"

    def build_data_chunks(self, url: str):
        loader = WebBaseLoader(url)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(data)

        for idx, chunk in enumerate(chunks):
            chunk.metadata["id"] = idx
        return chunks

    def get_embedding(self, chunk_list: list):
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.embedding_model_id}"
        headers = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}
        data = {"inputs": chunk_list, "options": {"wait_for_model": True}}
        response = requests.post(api_url, headers=headers, json=data)
        return response.json()


class Evaluator:
    """
    Evaluates query, context and responses using cosine similarity.
    """

    def __init__(self):
        self.embedding = EmbeddingUtility()

    def similarity(self, query: str, context: str, response: str):
        query_embedding = self.embedding.get_embedding(chunk_list=[query])[0]
        context_embedding = self.embedding.get_embedding(chunk_list=[context])[0]
        response_embedding = self.embedding.get_embedding(chunk_list=[response])[0]

        # Compute cosine similarity
        query_response_similarity = cosine_similarity(
            [query_embedding], [response_embedding]
        )[0][0]
        response_context_similarity = cosine_similarity(
            [response_embedding], [context_embedding]
        )[0][0]

        print("===== SIMILARITY =====")
        print(f"Query <> Response similarity : {query_response_similarity}")
        print(f"Response <> Context similarity : {response_context_similarity}")


class ChomaDBUtility:
    """
    Stores content (vector embeddings) in Chroma DB.
    Creates local file (.sqlite3) for storing data temporarily.
    """

    def __init__(self):
        self.embedding = EmbeddingUtility()
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="wikipedia_chunks"
        )

    def write_to_chroma(self, chunks: list):
        chunk_list = [chunk.page_content for chunk in chunks]
        embedding = self.embedding.get_embedding(chunk_list=chunk_list)

        # Store in ChromaDB
        for index, chunk in enumerate(chunks):
            self.collection.add(
                ids=[str(chunk.metadata["id"])],  # Unique ID
                embeddings=[embedding[index]],  # Vector representation
                documents=[chunk.page_content],  # Store original text
            )

        print(f"[INFO] Stored {len(chunks)} chunks in ChromaDB.")

    def read_from_chroma(self, query: str):
        embedding = self.embedding.get_embedding(chunk_list=[query])[0]
        results = self.collection.query(
            query_embeddings=[embedding], n_results=1  # Number of top matches
        )
        res = results["documents"][0][0]
        return res


# DRIVER METHODS


def load_data_to_chromadb():
    """
    Helper function to load wikipedia / any url data to Chroma DB.
    """
    chromadb_instance = ChomaDBUtility()
    embedding_instance = EmbeddingUtility()

    url = "https://en.wikipedia.org/wiki/Zomato"
    chunks = embedding_instance.build_data_chunks(url)
    # clean chunks [extract useful data only for testing]
    chunks = chunks[4:81]
    chromadb_instance.write_to_chroma(chunks=chunks)


def run():
    """
    Driver function for testing.
    """
    genai_instance = GenUtility()
    chromadb_instance = ChomaDBUtility()
    evaluator = Evaluator()

    # query = "which quick commerce company did zomato acquire?"
    query = "from info edge, how much did zomato raise fundings?"

    context = chromadb_instance.read_from_chroma(query=query)
    print("===== CHROMA DB RESULT =====")
    print(f"{context}\n")

    response = genai_instance.generate_response(query=query, context=context)

    evaluator.similarity(query=query, context=context, response=response)


run()
