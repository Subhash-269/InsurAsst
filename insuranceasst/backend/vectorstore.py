# backend/vectorstore.py
import os
import pickle
import re
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
        chunk_size: int = 800,
        chunk_overlap: int = 150,
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
            model=self.model,
        )
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        metadatas: List[Dict[str, Any]] = []
        for ch in chunks:
            src, page = None, None
            if getattr(ch, "metadata", None):
                # common locations used by loaders
                src = ch.metadata.get("source") or ch.metadata.get("file_path")
                page = ch.metadata.get("page")  # 0-based, set by PyPDFLoader
            metadatas.append(
                {
                    "text": ch.page_content,
                    "source": (src or "unknown"),
                    "page": page,
                    "page_end": ch.metadata.get("page_end", page) if getattr(ch, "metadata", None) else page,
                    "section": ch.metadata.get("section", "") if getattr(ch, "metadata", None) else "",
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

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    def _bm25(self):
        """Keyword index over section label + text, built lazily and rebuilt when the metadata changes."""
        if getattr(self, "_bm25_for", None) is not self.metadata:
            from rank_bm25 import BM25Okapi
            corpus = [self._tokens(f"{m.get('section', '')} {m.get('text', '')}") for m in self.metadata]
            self._bm25_index = BM25Okapi(corpus) if corpus else None
            self._bm25_for = self.metadata
        return self._bm25_index

    def query(self, query_text: str, top_k: int = 5, allowed_sources: Optional[List[str]] = None,
              hybrid: bool = False):
        """
        Top_k chunks for the query. If allowed_sources is provided (list of filenames), only those files are searched.
        hybrid=True fuses the vector ranking with a BM25 keyword ranking (reciprocal rank fusion).
        """
        print(f"[INFO] Querying vector store for: '{query_text}'")
        if self.index is None or self.index.ntotal == 0:
            return []
        query_emb = self.model.encode([query_text]).astype("float32")
        # the index is small, so rank everything and filter afterwards; a fixed over-retrieve could
        # be filled entirely by chunks from other files
        raw = self.search(query_emb, top_k=self.index.ntotal)

        allowed = {self._normalize_name(s) for s in allowed_sources} if allowed_sources else None
        def ok(r):
            return allowed is None or self._normalize_name((r.get("metadata") or {}).get("source", "")) in allowed
        dense = [r for r in raw if ok(r)]
        if not hybrid:
            return dense[:top_k]

        bm25 = self._bm25()
        scores = bm25.get_scores(self._tokens(query_text)) if bm25 else []
        keyword = sorted((i for i in range(len(self.metadata))
                          if ok({"metadata": self.metadata[i]}) and scores[i] > 0),
                         key=lambda i: -scores[i])
        fused: Dict[int, float] = {}
        for rank, r in enumerate(dense):
            fused[r["index"]] = fused.get(r["index"], 0) + 1 / (60 + rank)
        for rank, i in enumerate(keyword):
            fused[i] = fused.get(i, 0) + 1 / (60 + rank)
        best = sorted(fused, key=lambda i: -fused[i])[:top_k]
        return [{"index": i, "distance": -fused[i], "metadata": self.metadata[i]} for i in best]
