import os
from .data_loader import load_all_documents
from .vectorstore import FaissVectorStore
from .search__ import RAGSearch  

FAISS_DIR = "faiss_store"
FAISS_FILE = "faiss.index"
FAISS_PATH = os.path.join(FAISS_DIR, FAISS_FILE)

if __name__ == "__main__":
    # 1) Load documents
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")

    # 2) Prepare vector store
    store = FaissVectorStore(FAISS_DIR)

    # 3) Build the FAISS index if it doesn't exist; otherwise load it
    if not os.path.exists(FAISS_PATH):
        print(f"[INFO] No FAISS index found at {FAISS_PATH}. Building a new index...")
        store.build_from_documents(docs)
        # Persist the index to disk — adjust the method name if your class differs.
        if hasattr(store, "save"):
            store.save()
        elif hasattr(store, "persist"):
            store.persist()
        else:
            print("[WARN] Vector store has no save/persist method; ensure build_from_documents writes the index.")
    else:
        print(f"[INFO] Found existing FAISS index at {FAISS_PATH}. Loading...")
        store.load()

    # 4) Sanity check: if load() is the required step post-build, do it here too.
    try:
        if hasattr(store, "load"):
            store.load()
    except Exception as e:
        print(f"[WARN] load() after build raised: {e}")

    # 5) Test a query
    # results = store.query("What is attention mechanism?", top_k=3)
    # results = store.query("What should you do first if someone is injured in an auto accident?", top_k=3)

    # print(results)

    # 6) Optional: RAG pipeline
    # rag_search = RAGSearch()
    # query = "What is attention mechanism?"
    # summary = rag_search.search_and_summarize(query, top_k=3)
    # print("Summary:", summary)

    

    # Initialize the RAG pipeline with a local model
    # backend="ollama" uses a model pulled by Ollama (e.g., mistral, phi3, llama3)
    # backend="hf"     uses a Hugging Face model (e.g., microsoft/phi-3-mini-4k-instruct)
    rag_search = RAGSearch(
        persist_dir=FAISS_DIR,
        backend="ollama",       # or "hf"
        # model_name="phi3",      # e.g., "mistral", "llama3", "microsoft/phi-3-mini-4k-instruct"
        model_name="mistral",    
    )

    # Ask a question
    query = "What should you do first if someone is injured in an auto accident?"
    query = 'Does my policy cover rental cars or towing?'
    query = "Should I contact my insurance company first or wait for the police report?"
    query = "My car was damaged by a falling tree branch during a storm. Is this covered under my policy?" \
    " If so, what steps should I take to file a claim?"
    # query = "My car got a dent from a parking lot incident where the other driver left without leaving a note. Dont assume anything, let me know if need any more info"
    query = "My car's glass was shattered by a rock while driving on the highway. Does my insurance cover glass repairs or replacements?"
#     query = """
#        {
#   "questions": [
#     {
#       "id": "insurance_company_name",
#       "q": "What is the name of the issuing insurer/company on this policy?"
#     },
#     {
#       "id": "policy_type",
#       "q": "What is the type of insurance (e.g., Private Passenger Automobile Insurance Policy)?"
#     },
#     {
#       "id": "jurisdiction",
#       "q": "What jurisdiction/state does this policy apply to?"
#     },
#     {
#       "id": "total_coverage",
#       "q": "Summarize the headline minimum/standard coverage limits in dollars (e.g., Bodily Injury per person/per accident, Property Damage minimum, and any fixed PIP cap if explicitly stated). Return numbers only."
#     },
#     {
#       "id": "total_deductible",
#       "q": "What deductible amounts are explicitly stated (e.g., default Collision and Comprehensive deductibles)? Return numbers only."
#     }
#   ]
# }


# 
# """
    summary = rag_search.search_and_summarize(query, top_k=10)

    print("\n=== RAG Summary ===")
    print(summary)
