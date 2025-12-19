import os

MODEL_NAME = 'llama3.2'
EMBEDDING_MODEL = 'nomic-embed-text' #ollama pull nomic-embed-text

DATA_DIR = 'data'
STORAGE_DIR = 'storage'

CHUNK_SIZE = 1024
RETRIEVAL_TOP_K = 3

PROMPT_TEMPLATE = """

you are an AI history agent
make sure to answer all the questions from the **given document**

if the answer is not available in the provided data, you must state 
"i don't have answet to this particular 

context:
{context}

question:
{question}

answer in 150 words.

"""

os.makedirs('data', exist_ok=True)
os.makedirs('storage', exist_ok=True)