import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from backend.vectorstore import FaissVectorStore           # ✅ single, correct import
from backend.data_loader import load_all_documents         # ✅ single, correct import

load_dotenv()

# ---- Local LLM backends ------------------------------------------------------

class LocalLLM:
    """Unified interface for local LLM backends."""
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        raise NotImplementedError


class OllamaLLM(LocalLLM):
    """Uses the local Ollama server."""
    def __init__(self, model: str = "phi3", host: Optional[str] = None):
        import ollama
        self.ollama = ollama
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", None)
        if self.host:
            os.environ["OLLAMA_HOST"] = self.host

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        resp = self.ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": max_new_tokens,
                "temperature": 0.2,
                "top_p": 0.95,
            },
        )
        return resp["message"]["content"].strip()


class HFTransformersLLM(LocalLLM):
    """Runs a HF model locally (CPU or GPU)."""
    def __init__(
        self,
        model_id: str = "microsoft/phi-3-mini-4k-instruct",
        device: Optional[str] = None,
        load_4bit: bool = True,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        quant_cfg = BitsAndBytesConfig(load_in_4bit=True) if load_4bit else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if device is None else None,
            quantization_config=quant_cfg,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device is not None:
            self.model.to(device)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text.strip()

# ---- RAG wrapper -------------------------------------------------------------

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        backend: str = "ollama",                # "ollama" or "hf"
        model_name: str = "phi3",               # Ollama model name OR HF model_id
        hf_4bit: bool = True,
    ):
        """
        backend="ollama" -> model_name like: "phi3", "mistral", "llama3", "gemma", "tinyllama"
        backend="hf"     -> model_name is a HF repo id, e.g. "microsoft/phi-3-mini-4k-instruct"
        """
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        # Load or (if missing) build FAISS index
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path  = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        # Choose local LLM backend
        if backend.lower() == "ollama":
            self.llm: LocalLLM = OllamaLLM(model=model_name)
            print(f"[INFO] Using Ollama model: {model_name}")
        elif backend.lower() == "hf":
            self.llm = HFTransformersLLM(model_id=model_name, load_4bit=hf_4bit)
            print(f"[INFO] Using HF Transformers model: {model_name}")
        else:
            raise ValueError("backend must be 'ollama' or 'hf'")

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            "You are an insurance FAQ assistant using retrieved context.\n"
            "Answer briefly and factually. If unsure, say so.\n\n"
            f"Query:\n{query}\n\nContext:\n{context}\n\nAnswer:"
        )

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5,
        max_new_tokens: int = 256,
        doc: Optional[str] = None,
    ) -> str:
        # Use allowed_sources when doc is provided (requires vectorstore.query to support it)
        results: List[Dict[str, Any]] = self.vectorstore.query(
            query,
            top_k=top_k,
            allowed_sources=[doc] if doc else None
        )
        texts = [r.get("metadata", {}).get("text", "") for r in results if r.get("metadata")]

        # de-dup + truncate
        uniq, seen = [], set()
        for t in texts:
            t2 = t.strip()
            if t2 and t2 not in seen:
                seen.add(t2)
                uniq.append(t2)

        context = "\n\n---\n\n".join(uniq)[:8000]
        if not context:
            return "No relevant documents found for the selected source(s)."

        prompt = self._build_prompt(query, context)
        return self.llm.generate(prompt, max_new_tokens=max_new_tokens)


if __name__ == "__main__":
    rag = RAGSearch(backend="ollama", model_name="phi3")
    query = "What is the attention mechanism?"
    summary = rag.search_and_summarize(query, top_k=3, max_new_tokens=256)
    print("Summary:", summary)
