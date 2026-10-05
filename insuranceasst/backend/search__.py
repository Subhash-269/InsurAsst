import os
import re
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from backend.vectorstore import FaissVectorStore           # ✅ single, correct import
from backend.data_loader import load_all_documents         # ✅ single, correct import

load_dotenv()

# ---- Local LLM backends ------------------------------------------------------

Messages = List[Dict[str, str]]  # [{"role": "system"|"user"|"assistant", "content": ...}]


class LocalLLM:
    """Unified interface for local LLM backends."""
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        return self.chat([{"role": "user", "content": prompt}], max_new_tokens=max_new_tokens)

    def chat(self, messages: Messages, max_new_tokens: int = 256) -> str:
        raise NotImplementedError

    def stream_chat(self, messages: Messages, max_new_tokens: int = 512):
        """Default: generate the full text, then yield it word by word."""
        for piece in self.chat(messages, max_new_tokens=max_new_tokens).split():
            yield piece + " "


class OllamaLLM(LocalLLM):
    """Uses the local Ollama server."""
    # Ollama loads Mistral with a 4096-token window by default; passages + history + answer need more
    NUM_CTX = 8192
    seed = 42  # fixed so the same question gets the same answer (evaluation runs stay comparable)

    def __init__(self, model: str = "phi3", host: Optional[str] = None):
        import ollama
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", None)
        self.client = ollama.Client(host=self.host)

    def _options(self, max_new_tokens: Optional[int]) -> Dict[str, Any]:
        # fixed seed so the same question gets the same answer (makes evaluation runs comparable)
        options = {"temperature": 0.2, "top_p": 0.95, "num_ctx": self.NUM_CTX, "seed": self.seed}
        if max_new_tokens:
            options["num_predict"] = max_new_tokens
        return options

    def chat(self, messages: Messages, max_new_tokens: int = 256) -> str:
        resp = self.client.chat(model=self.model, messages=messages, options=self._options(max_new_tokens))
        return resp["message"]["content"].strip()

    def stream_chat(self, messages: Messages, max_new_tokens: Optional[int] = None):
        """Yield the answer in chunks as Ollama produces them."""
        for part in self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options=self._options(max_new_tokens),
        ):
            chunk = (part.get("message") or {}).get("content") or ""
            if chunk:
                yield chunk


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

    def chat(self, messages: Messages, max_new_tokens: int = 256) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages) + "\n\nASSISTANT:\n"
        return self.generate(prompt, max_new_tokens=max_new_tokens)

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
        # decode only the newly generated tokens, not the echoed prompt
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
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
        data_dir: str = "data",
        hybrid: bool = False,
        expand_query: bool = False,
        neighbors: bool = False,
        route: bool = False,
    ):
        """
        backend="ollama" -> model_name like: "phi3", "mistral", "llama3", "gemma", "tinyllama"
        backend="hf"     -> model_name is a HF repo id, e.g. "microsoft/phi-3-mini-4k-instruct"
        hybrid           -> fuse keyword (BM25) and vector search
        expand_query     -> ask the LLM to translate the customer's story into policy terms before searching
        neighbors        -> extend each retrieved chunk with the next chunk of the same section
        route            -> let the LLM pick the relevant coverage sections from a catalogue of the policy first
        """
        self.data_dir = data_dir
        self.hybrid = hybrid
        self.expand_query = expand_query
        self.neighbors = neighbors
        self.route = route
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        self.reload_index()

        # Choose local LLM backend
        if backend.lower() == "ollama":
            self.llm: LocalLLM = OllamaLLM(model=model_name)
            print(f"[INFO] Using Ollama model: {model_name}")
        elif backend.lower() == "hf":
            self.llm = HFTransformersLLM(model_id=model_name, load_4bit=hf_4bit)
            print(f"[INFO] Using HF Transformers model: {model_name}")
        else:
            raise ValueError("backend must be 'ollama' or 'hf'")

    def reload_index(self):
        """Load the FAISS index from disk, building it from data_dir if it is missing or empty."""
        persist_dir = self.vectorstore.persist_dir
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path  = os.path.join(persist_dir, "metadata.pkl")
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.vectorstore.load()
            return
        # reset in-memory state so a rebuild doesn't append to stale vectors
        self.vectorstore.index = None
        self.vectorstore.metadata = []
        docs = load_all_documents(str(self.data_dir))
        if docs:
            self.vectorstore.build_from_documents(docs)

    def retrieve(self, query: str, top_k: int = 5, doc: Optional[str] = None) -> List[Dict[str, Any]]:
        """Top_k de-duplicated chunks (optionally limited to one source file) as {text, source, page}."""
        if self.vectorstore.index is None:
            return []
        results: List[Dict[str, Any]] = self.vectorstore.query(
            query,
            top_k=top_k,
            allowed_sources=[doc] if doc else None,
            hybrid=self.hybrid,
        )
        hits, used = [], set()
        for r in results:
            hit = self._make_hit(r.get("index", -1), used)
            if hit:
                hits.append(hit)
        return hits

    def _make_hit(self, idx: int, used: set) -> Optional[Dict[str, Any]]:
        """Chunk idx as a passage (extended with the next chunk of its section when neighbors is on)."""
        md = self.vectorstore.metadata
        if not (0 <= idx < len(md)) or idx in used:
            return None
        meta = md[idx]
        text = (meta.get("text") or "").strip()
        if not text:
            return None
        used.add(idx)
        section = meta.get("section", "")
        page_end = meta.get("page_end", meta.get("page"))
        if self.neighbors:
            nxt = self._next_in_section(idx)
            if nxt is not None and idx + 1 not in used:
                used.add(idx + 1)
                text = self._join_overlapping(text, nxt.get("text") or "")
                page_end = nxt.get("page_end", nxt.get("page"))
                section = " | ".join(dict.fromkeys(filter(None, section.split(" | ") + nxt.get("section", "").split(" | "))))
        return {"text": text, "source": meta.get("source", ""), "page": meta.get("page"), "page_end": page_end,
                "section": section}

    def _section_hit(self, entry: Dict[str, Any], used: set, min_chars: int = 1400) -> Optional[Dict[str, Any]]:
        """A routed section as one passage: starts at its heading (not at the start of the chunk that
        happens to contain it) and runs on into the following chunks until it has enough text."""
        md = self.vectorstore.metadata
        i = entry["index"]
        if not (0 <= i < len(md)) or i in used:
            return None
        first = md[i]
        text = (first.get("text") or "")[entry.get("offset", 0):].strip()
        page_end = first.get("page_end", first.get("page"))
        used.add(i)
        j = i + 1
        while len(text) < min_chars and j < len(md) and md[j].get("source") == first.get("source"):
            text = self._join_overlapping(text, md[j].get("text") or "")
            page_end = md[j].get("page_end", md[j].get("page"))
            used.add(j)
            j += 1
        return {"text": text, "source": first.get("source", ""), "page": first.get("page"), "page_end": page_end,
                "section": entry["label"]}

    # ---- Coverage routing ------------------------------------------------------
    # Which coverage applies is a question about the whole policy, not about wording similarity:
    # "someone hit my car and drove off" reads like the Uninsured Motorists hit-and-run text, but the
    # car damage falls under Collision. So the LLM first picks sections from a catalogue of the policy.

    _SKIP_SECTIONS = ("insured persons", "insured autos", "definitions", "limits of liability",
                      "payment of loss", "right to appraisal")

    def catalog(self, doc: Optional[str]) -> List[Dict[str, Any]]:
        """Sections of one policy file: label, page, first chunk index and opening words."""
        md = self.vectorstore.metadata
        if getattr(self, "_catalog_for", None) is not md:
            self._catalogs, self._catalog_for = {}, md
        key = (doc or "").lower()
        if key in self._catalogs:
            return self._catalogs[key]
        entries, seen = [], set()
        for i, m in enumerate(md):
            if doc and os.path.basename(m.get("source") or "").lower() != key:
                continue
            for label in filter(None, (m.get("section") or "").split(" | ")):
                tail = label.split(" / ")[-1]
                if label in seen or tail.lower().startswith(self._SKIP_SECTIONS):
                    continue
                seen.add(label)
                text = " ".join((m.get("text") or "").split())
                probe = " ".join(tail.split()[:2]).lower()
                pos = text.lower().find(probe)
                opening = text[pos if pos >= 0 else 0:]
                # same heading position in the raw chunk text (line breaks intact), for _section_hit
                raw_pos = re.search(r"\s+".join(map(re.escape, probe.split())), (m.get("text") or "").lower())
                offset = raw_pos.start() if raw_pos else 0
                if len(opening) < 220 and i + 1 < len(md) and md[i + 1].get("source") == m.get("source"):
                    # heading near the end of a chunk: the definition continues in the next one
                    nxt = " ".join((md[i + 1].get("text") or "").split())
                    tail_at = nxt.find(opening[-60:])
                    opening += nxt[tail_at + len(opening[-60:]):] if tail_at >= 0 else " " + nxt
                entries.append({"label": label, "page": (m.get("page") or 0) + 1, "index": i,
                                "offset": offset, "opening": opening[:220]})
        # a Part's own introduction is only useful when the Part has no separate coverages
        labels = [e["label"] for e in entries]
        entries = [e for e in entries
                   if not any(other.startswith(e["label"] + " / Coverage") for other in labels)]
        self._catalogs[key] = entries
        return entries

    ROUTE_PROMPT = (
        "You help find the right sections of a customer's auto insurance policy. You get a numbered list of "
        "the policy's coverages and sections, each with its opening words, then the customer's situation.\n"
        "Pick the sections needed to answer: first the specific coverage whose own definition fits the loss, "
        "then at most one section on duties after a loss, exclusions or additional payments if it matters.\n"
        "Check what each coverage pays for. Damage to the customer's own car falls under collision-type or "
        "comprehensive-type coverages. Sections that pay for injuries or for damage the customer causes to "
        "other people or their property do not pay for the customer's own car.\n"
        "Reply with 1 to 3 numbers, most relevant first, comma-separated, and nothing else."
    )

    def route_sections(self, query: str, doc: Optional[str]) -> List[Dict[str, Any]]:
        entries = self.catalog(doc)
        if not entries:
            return []
        listing = "\n".join(f"{n}. {e['label']} (p.{e['page']}): {e['opening']}" for n, e in enumerate(entries, 1))
        try:
            reply = self.llm.chat(
                [{"role": "system", "content": self.ROUTE_PROMPT},
                 {"role": "user", "content": f"Sections:\n{listing}\n\nCustomer situation:\n{query}"}],
                max_new_tokens=20,
            )
        except Exception as e:
            print(f"[WARN] routing failed: {e}")
            return []
        picked = []
        for n in re.findall(r"\d+", reply):
            k = int(n) - 1
            if 0 <= k < len(entries) and entries[k] not in picked:
                picked.append(entries[k])
        picked = picked[:3]
        # an adjuster always checks a coverage's exclusions: add the Exclusions section of the routed Part
        for e in list(picked):
            if " / Coverage" in e["label"]:
                part = e["label"].split(" / ")[0]
                excl = next((x for x in entries if x["label"].startswith(f"{part} / Exclusions")), None)
                if excl and excl not in picked:
                    picked.append(excl)
                break
        print(f"[INFO] Routed to: {[e['label'] for e in picked]}")
        return picked

    def _next_in_section(self, idx: int) -> Optional[Dict[str, Any]]:
        """The chunk that follows idx, if it continues the same section of the same file."""
        md = self.vectorstore.metadata
        if not (0 <= idx < len(md) - 1):
            return None
        cur, nxt = md[idx], md[idx + 1]
        if cur.get("source") != nxt.get("source"):
            return None
        cur_labels = set(filter(None, (cur.get("section") or "").split(" | ")))
        nxt_first = (nxt.get("section") or "").split(" | ")[0]
        return nxt if nxt_first and nxt_first in cur_labels else None

    @staticmethod
    def _join_overlapping(a: str, b: str) -> str:
        """Concatenate consecutive chunks without repeating the splitter's overlap."""
        probe = b[:80]
        pos = a.find(probe) if probe else -1
        return a[:pos] + b if pos != -1 else f"{a}\n{b}"

    @staticmethod
    def context_from(hits: List[Dict[str, Any]], max_chars: int = 9000) -> str:
        """Numbered passages; the numbers match the UI's Sources panel so the model can cite [n]."""
        parts, used = [], 0
        for i, h in enumerate(hits, 1):
            page, page_end = h.get("page"), h.get("page_end")
            where = os.path.basename(h.get("source") or "")
            if isinstance(page, int):
                where += f", p.{page + 1}" + (f"-{page_end + 1}" if isinstance(page_end, int) and page_end > page else "")
            if h.get("section"):
                where += f" — {h['section']}"
            block = f"[{i}] ({where})\n{h['text']}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    @staticmethod
    def sources_from(hits: List[Dict[str, Any]], snippet_chars: int = 600) -> List[Dict[str, Any]]:
        """Compact citations for the UI: file name, 1-based page (if known) and a passage snippet."""
        out = []
        for h in hits:
            page, page_end = h.get("page"), h.get("page_end")
            out.append({
                "name": os.path.basename(h.get("source") or ""),
                "page": page + 1 if isinstance(page, int) else None,
                "page_end": page_end + 1 if isinstance(page_end, int) else None,
                "section": h.get("section", ""),
                "snippet": " ".join(h["text"].split())[:snippet_chars],
            })
        return out

    def build_context(self, query: str, top_k: int = 5, doc: Optional[str] = None) -> str:
        """Retrieve top_k chunks (optionally limited to one source file), de-dup and truncate."""
        return self.context_from(self.retrieve(query, top_k=top_k, doc=doc))

    SYSTEM_PROMPT = (
        "You are an auto-insurance claims assistant. A customer describes what happened and you explain what "
        "their own policy says. Each user turn includes numbered policy passages.\n\n"
        "How to answer:\n"
        "1. Decide which coverage fits by checking each coverage's own definition in the passages against the "
        "facts. A coverage applies only if its definition fits the loss. For example, a coverage that pays for "
        "bodily injury does not pay for damage to the car, and a coverage that won't pay when the other driver "
        "can't be identified does not pay for a hit-and-run.\n"
        "2. Use only the passages and what the customer told you. Never invent coverages, amounts, time limits "
        "or conditions, and don't rely on general insurance knowledge.\n"
        "3. State key conditions precisely: amounts, days, deadlines, and anything that 'does not apply'. "
        "Do not reverse or soften them.\n"
        "4. Cite the passage number for every policy statement, like [2]. Only cite numbers that exist.\n"
        "5. Which coverages the customer bought, and their deductibles and limits, are on their Declarations "
        "or Coverage Selections page, which you can't see. Say 'if you have collision coverage' rather than "
        "assuming.\n"
        "6. Refer to the insurer in the third person ('Allstate will pay', 'your insurer'), never as 'we'.\n"
        "7. If the passages don't address the question, say so plainly and don't guess.\n\n"
        "Format:\n"
        "**Short answer:** one or two sentences.\n"
        "**What your policy says:** 2-4 bullets with citations.\n"
        "**What to do next:** bullets, only if the passages give steps or deadlines.\n"
        "**To give you a precise answer:** 1-3 numbered questions, only if the answer depends on facts you "
        "don't have.\n\n"
        "Good questions ask for facts only the customer knows that would change the answer: which coverages "
        "are on their Declarations page (for example collision, comprehensive or rental), whether anyone was "
        "injured, whether the police were notified, when it happened, whether the car can be driven. Never ask "
        "the customer what the policy says, never ask about something they already told you, and don't ask for "
        "numbers they would have to look up (limits, deductible amounts) unless the answer depends on them. "
        "When you have what you need, leave this section out entirely; never write that no questions are needed."
    )

    @staticmethod
    def retrieval_query(question: str, history: Optional[Messages] = None) -> str:
        """Short follow-ups ('what about a rental?') need the previous question to retrieve well."""
        prev = [m["content"] for m in (history or []) if m.get("role") == "user"]
        return f"{prev[-1]}\n{question}" if prev else question

    def build_messages(
        self,
        question: str,
        hits: List[Dict[str, Any]],
        history: Optional[Messages] = None,
        facts: Optional[str] = None,
    ) -> Messages:
        system = self.SYSTEM_PROMPT
        if facts:
            system += f"\n\nKnown facts from this conversation:\n{facts}"
        msgs: Messages = [{"role": "system", "content": system}]
        # earlier turns carry the conversation; only the current turn carries passages (keeps the prompt small)
        for m in history or []:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({
            "role": "user",
            "content": f"Policy passages:\n\n{self.context_from(hits)}\n\nCustomer: {question}",
        })
        return msgs

    def prepare(
        self,
        question: str,
        doc: Optional[str] = None,
        history: Optional[Messages] = None,
        facts: Optional[str] = None,
        top_k: int = 5,
    ):
        """Retrieve passages and build the chat messages for one turn. Shared by the chat view and the evaluator."""
        query = self.retrieval_query(question, history)
        if self.expand_query:
            query = f"{query}\n{self.policy_terms(query)}"
        hits = self.retrieve(query, top_k=top_k, doc=doc)
        if self.route:
            # routed sections first, then the best search hits, without repeating text
            used: set = set()
            routed = [h for e in self.route_sections(query, doc) if (h := self._section_hit(e, used))]
            seen = [h["text"] for h in routed]
            extra = [h for h in hits if not any(h["text"][:300] in s or s[:300] in h["text"] for s in seen)]
            hits = (routed + extra)[:max(top_k, 6)]
        hits = self._fit(hits)
        return hits, self.build_messages(question, hits, history, facts)

    def _fit(self, hits: List[Dict[str, Any]], max_chars: int = 9000) -> List[Dict[str, Any]]:
        """Keep only the passages that fit in the prompt, so the Sources panel matches what the model saw."""
        kept, used = [], 0
        for h in hits:
            used += len(h["text"]) + 120  # + passage header
            if used > max_chars and kept:
                break
            kept.append(h)
        return kept

    EXPAND_PROMPT = (
        "You turn a customer's description of a car-insurance situation into search terms for finding the "
        "relevant sections of their auto policy. Reply with one line of 6-12 comma-separated terms: the kinds "
        "of coverage that could apply, the type of loss, and duties or conditions that matter. Use policy "
        "vocabulary, for example: collision, comprehensive, glass breakage, theft, towing and labor, rental "
        "reimbursement, substitute transportation, uninsured motorists bodily injury, medical payments, "
        "deductible, exclusions, mechanical breakdown, wear and tear, what you must do if there is a loss, "
        "notify police, proof of loss. No explanation."
    )

    def policy_terms(self, query: str) -> str:
        """One short LLM call that maps everyday wording ('drove off', 'transmission died') to policy terms."""
        try:
            terms = self.llm.chat(
                [{"role": "system", "content": self.EXPAND_PROMPT}, {"role": "user", "content": query}],
                max_new_tokens=60,
            )
            return " ".join(terms.split())[:300]
        except Exception as e:
            print(f"[WARN] query expansion failed: {e}")
            return ""

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5,
        max_new_tokens: int = 512,
        doc: Optional[str] = None,
        history: Optional[Messages] = None,
    ) -> str:
        hits = self.retrieve(self.retrieval_query(query, history), top_k=top_k, doc=doc)
        if not hits:
            return "No relevant documents found for the selected source(s)."
        return self.llm.chat(self.build_messages(query, hits, history), max_new_tokens=max_new_tokens)


if __name__ == "__main__":
    rag = RAGSearch(backend="ollama", model_name="phi3")
    query = "What is the attention mechanism?"
    summary = rag.search_and_summarize(query, top_k=3, max_new_tokens=256)
    print("Summary:", summary)
