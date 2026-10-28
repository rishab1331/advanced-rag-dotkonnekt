import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Free Hosted LLM API (Groq - OpenAI compatible)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    LLM_MODEL = "openai/gpt-oss-120b"
    
    # Free Local Embedding Model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Path to your 15 .txt files
    DOCS_DIR = "documents"
    
    # Chunking settings
    CHUNK_SIZE_WORDS = 180
    CHUNK_OVERLAP_WORDS = 35
    
    # Hybrid Retrieval & RRF settings
    TOP_K_VECTOR = 6
    TOP_K_BM25 = 6
    RRF_K = 60
    RERANK_TOP_K = 4