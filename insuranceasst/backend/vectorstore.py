# backend/vectorstore.py
import os
import pickle
from typing import List, Any, Dict, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []

        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"[INFO] Loaded embedding model: {embedding_model}")

    # ---------- Build ----------
    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        metadatas: List[Dict[str, Any]] = []
        for ch in chunks:
            src = None
            if getattr(ch, "metadata", None):
                # common locations used by loaders
                src = ch.metadata.get("source") or ch.metadata.get("file_path")
            metadatas.append(
                {
                    "text": ch.page_content,
                    "source": (src or "unknown"),
                }
            )

        self.add_embeddings(np.asarray(embeddings, dtype="float32"), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)  # simple, fast L2 index
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    # ---------- Persistence ----------
    def save(self):
        if self.index is None:
            raise RuntimeError("No index to save. Did you build/add embeddings?")
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(
                f"Missing index files in {self.persist_dir}. Build the index first."
            )
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    # ---------- Query ----------
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() or build_from_documents() first.")
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if 0 <= idx < len(self.metadata) else {}
            results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results

    def _normalize_name(self, s: str) -> str:
        return (os.path.basename(s) if s else "").lower()

    def query(self, query_text: str, top_k: int = 5, allowed_sources: Optional[List[str]] = None):
        """
        If allowed_sources is provided (list of filenames), only return hits from those files.
        """
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype("float32")

        # over-retrieve so filtering still returns enough
        raw = self.search(query_emb, top_k=max(top_k * 5, 20))

        if not allowed_sources:
            return raw[:top_k]

        allowed = {self._normalize_name(s) for s in allowed_sources}
        filtered = []
        for r in raw:
            meta = r.get("metadata") or {}
            src = self._normalize_name(meta.get("source", ""))
            if src in allowed:
                filtered.append(r)
            if len(filtered) >= top_k:
                break
        return filtered
