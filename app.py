# app.py
# Simple Streamlit interface for the MyScheme RAG QA system

import streamlit as st
import faiss
import torch
import json
import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Configuration (Should match the RAG script) ---
FAISS_INDEX_FILE = "my_scheme_faiss.index"
METADATA_FILE = "my_scheme_metadata.json"
CHUNKS_FILE = "my_scheme_chunks.json"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
GENERATOR_MODEL_NAME = "google/flan-t5-base"
RAG_TOP_K = 5
LOGGING_LEVEL = logging.INFO

# Configure basic logging for the app
logging.basicConfig(level=LOGGING_LEVEL, format='%(asctime)s [%(levelname)-8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

# --- Caching Function to Load Models and Data ---
# @st.cache_resource ensures these heavy components are loaded only once.
@st.cache_resource
def load_rag_components():
    """Loads the FAISS index, metadata, chunks, and ML models."""
    log.info("--- Loading RAG Components ---")
    loaded_components = {}
    files_ok = True

    # Check for files
    required_files = [FAISS_INDEX_FILE, METADATA_FILE, CHUNKS_FILE]
    if not all(os.path.exists(f) for f in required_files):
        st.error(f"Error: Missing one or more data files: {required_files}. Please run the main RAG processing script first.")
        log.critical(f"Missing RAG data files: {required_files}")
        return None

    try:
        # Load data files
        log.info(f"Loading FAISS index from {FAISS_INDEX_FILE}...")
        loaded_components['index'] = faiss.read_index(FAISS_INDEX_FILE)
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            loaded_components['chunk_metadata'] = json.load(f)
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            loaded_components['chunks'] = json.load(f)
        log.info("Index, metadata, and chunks loaded.")

        # Verify counts
        if not (loaded_components['index'].ntotal == len(loaded_components['chunk_metadata']) == len(loaded_components['chunks'])):
             log.warning("Loaded data count mismatch!")
             # Optionally add st.warning here

        # Load embedding model
        log.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Using device: {device} for embeddings")
        loaded_components['embedding_model'] = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        log.info("Embedding model loaded.")

        # Load generator model
        log.info(f"Loading generator model: {GENERATOR_MODEL_NAME}...")
        loaded_components['generator_tokenizer'] = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
        loaded_components['generator_model'] = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL_NAME, device_map="auto")
        loaded_components['generator_model'].eval()
        log.info(f"Generator model ready on device(s): {loaded_components['generator_model'].device_map if hasattr(loaded_components['generator_model'], 'device_map') else next(loaded_components['generator_model'].parameters()).device}")

        log.info("--- All RAG Components Loaded Successfully ---")
        return loaded_components

    except Exception as e:
        log.critical(f"Failed to load RAG components: {e}", exc_info=True)
        st.error(f"Error loading RAG components: {e}")
        return None

# --- RAG Functions (Adapted from the previous script) ---

def retrieve_relevant_chunks_app(query: str, model, faiss_index, chunk_list, metadata_list, k: int = RAG_TOP_K):
    """Retrieval function adapted for the Streamlit app."""
    log.debug(f"Retrieving top {k} for query: '{query[:60]}...'")
    try:
        query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        scores, indices = faiss_index.search(query_embedding, k)
        results = []
        if indices.size > 0:
             for i, idx in enumerate(indices[0]):
                  if idx != -1 and 0 <= idx < len(chunk_list):
                       results.append({"chunk_text": chunk_list[idx], "metadata": metadata_list[idx], "score": float(scores[0][i])})
        results.sort(key=lambda x: x['score'], reverse=True)
        log.debug(f"Retrieved {len(results)} chunks.")
        return results
    except Exception as e: log.error(f"Retrieval error in app: {e}", exc_info=True); return []

def generate_answer_app(query: str, context_chunk_texts: list, model, tokenizer):
    """Generation function adapted for the Streamlit app."""
    if not context_chunk_texts: return "I couldn't find relevant information..."
    context_string = "\n\n".join(context_chunk_texts)
    prompt_template = """Please answer the following question based *only* on the provided context information. If the context does not contain the answer, please state 'Based on the provided context, I cannot answer this question'.

Context:
---
{context}
---

Question: {question}

Answer:"""
    prompt = prompt_template.format(context=context_string, question=query)
    log.debug(f"Generator prompt (start): {prompt[:300]}...")
    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4, early_stopping=True, no_repeat_ngram_size=2)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        log.debug(f"Generated answer: {answer[:100]}...")
        return answer.strip()
    except Exception as e: log.error(f"Generation error in app: {e}", exc_info=True); return "Error during answer generation."

# --- Streamlit App Interface ---

st.set_page_config(page_title="MyScheme QA Bot", layout="wide")
st.title("🇮🇳 MyScheme QA Bot")
st.caption("Ask questions about Indian government schemes based on scraped data.")

# Load components using caching
rag_components = load_rag_components()

if rag_components:
    # Create input field for user query
    user_query = st.text_input("Ask your question about a scheme:", placeholder="e.g., What is the benefit amount for the PM Scholarship for RPF?")

    # Add a button to trigger the query
    if st.button("Get Answer"):
        if user_query:
            log.info(f"Streamlit Query Received: \"{user_query}\"")
            # Use a spinner while processing
            with st.spinner("Retrieving relevant info and generating answer..."):
                # 1. Retrieve context
                retrieved = retrieve_relevant_chunks_app(
                    query=user_query,
                    model=rag_components['embedding_model'],
                    faiss_index=rag_components['index'],
                    chunk_list=rag_components['chunks'],
                    metadata_list=rag_components['chunk_metadata'],
                    k=RAG_TOP_K
                )

                # 2. Generate answer
                if not retrieved:
                    answer = "I couldn't find specific information related to your question in the available scheme data."
                else:
                    context_texts = [r['chunk_text'] for r in retrieved]
                    answer = generate_answer_app(
                        query=user_query,
                        context_chunk_texts=context_texts,
                        model=rag_components['generator_model'],
                        tokenizer=rag_components['generator_tokenizer']
                    )

                # Display the answer
                st.markdown("### Answer:")
                st.info(answer) # Use info box for the answer

                # Optional: Display retrieved context for debugging/transparency
                with st.expander("Show Retrieved Context Snippets (for debugging)"):
                    if retrieved:
                        for i, r in enumerate(retrieved):
                            st.markdown(f"**Chunk {i+1} (Score: {r['score']:.4f})**")
                            st.caption(f"Source: {r['metadata'].get('scheme_name','?')}, Field: {r['metadata'].get('field','?')}")
                            st.markdown(f"> {r['chunk_text'][:350]}...") # Show snippet
                    else:
                        st.write("No context was retrieved.")

        else:
            st.warning("Please enter a question.")
else:
    st.error("RAG system could not be initialized. Please check logs and ensure data/index files are present.")

st.markdown("---")
st.caption("Powered by Flan-T5, BGE Embeddings, FAISS, and Streamlit.")