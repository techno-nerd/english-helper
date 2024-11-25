from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
import pickle
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter



CHUNK_SIZE = 512
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
VECTOR_PATH = "vectors/vector_store.index"
METADATA_PATH = "vectors/metadata.pkl"


@st.cache_resource
def load_index(index_path=VECTOR_PATH, metadata_path=METADATA_PATH):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def create_index(pdf):
    pdf_reader = PdfReader(pdf)
    chunks = []
    metadata = []  # To store page_number, index and text for each chunk
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
            separators=[
                    "\n\n",
                    "\n",
                    ".",
                    ",",
                    " "
                    ]
            )

    #Saves chunks with corresponding information
    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        page_chunks = text_splitter.split_text(text)
        for chunk in page_chunks:
            chunks.append(chunk)
            metadata.append({"page_number": page_num + 1, 
                             "chunk_index": len(chunks) - 1,
                             "text":chunk})

    # Embed the text chunks
    embeddings = MODEL.encode(chunks)
    
    try:
        os.mkdir("vectors/")
    except:
        pass


    # Create and save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, VECTOR_PATH)

    # Save metadata
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Index created with {index.ntotal} vectors and metadata saved.")


def query_index(query, index, metadata, top_k=5):

    query_embedding = MODEL.encode([query])

    # Search the FAISS index
    distances, indices = index.search(np.array(query_embedding), k=top_k)

    # Retrieve matching chunks and metadata
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        result = metadata[idx]
        result["distance"] = dist
        results.append(result)

    #Sort them in ascending order of distance
    results = sorted(results, key=lambda x: x['distance'])
    return results