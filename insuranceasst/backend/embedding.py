import re
from collections import defaultdict
from typing import List, Any, Optional, Dict, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import numpy as np
from .data_loader import load_all_documents

# ---- Section detection ---------------------------------------------------------
# Policies are organised by Part and Coverage; tagging each chunk with its section lets both the
# search and the LLM tell "Collision" text from "Uninsured Motorists" text.

# table-of-contents line: "Part 7. Collision ........ 16" (MA) / "Part 5 __ Uninsured Motorists Insurance" (Allstate)
_TOC = re.compile(r"^Part (\d+)\.?\s*(?:__\s*(.+)|(.+?)\s*\.{3,})")
_TOC_ENTRY = re.compile(r"\.{3,}|\s\d+\s*$")   # dot leaders or a trailing page number
_TOP_LEVEL = ("general provisions", "when there is")  # sections that end the current Part
_PART_ALONE = re.compile(r"^Part (\d+)\.?\s*$")             # "Part 6" (Allstate) or "Part 8." (MA)
_PART_SENTENCE = re.compile(r"^Part (\d+)\.\s+\S")           # "Part 7. Under this Part, ..." (MA)
_COVERAGE = re.compile(r"^COVERAGE ([A-Z]{2})\s*$")          # "COVERAGE UU" + title on the next line
_GENERAL = re.compile(
    r"^(What You Must Do If There Is A Loss|When There Is|Exclusions\b|Additional Payments\b|Insured Autos\b|"
    r"Insured Persons\b|Definitions\b|Limits Of Liability|Right To Appraisal|Payment Of Loss|General Provisions)",
    re.I,
)


def _part_titles(lines: List[str]) -> Dict[str, str]:
    titles = {}
    for line in lines:
        m = _TOC.match(line.strip())
        if m and m.group(1) not in titles:
            titles[m.group(1)] = re.sub(r"\s+", " ", m.group(2) or m.group(3)).strip(" _.")
    return titles


def _section_events(pages: List[Tuple[int, str]]) -> Dict[int, List[Tuple[int, str]]]:
    """For each page index: [(char offset, section label)] where a new section starts."""
    all_lines = [ln for _, text in pages for ln in text.splitlines()]
    titles = _part_titles(all_lines)
    events: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    part, sub = "", ""
    for page_no, text in pages:
        lines = text.splitlines(keepends=True)
        offset = 0
        for i, raw in enumerate(lines):
            line = raw.strip()
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            label = None
            if _TOC.match(line):
                pass  # table of contents, not a section start
            elif (m := _PART_ALONE.match(line)) and m.group(1) in titles and (
                not line.endswith(".") or "Under this Part" in nxt
            ):
                part, sub = f"Part {m.group(1)} {titles[m.group(1)]}", ""
                label = part
            elif (m := _PART_SENTENCE.match(line)) and m.group(1) in titles and "...." not in line:
                part, sub = f"Part {m.group(1)} {titles[m.group(1)]}", ""
                label = part
            elif (m := _COVERAGE.match(line)):
                sub = f"Coverage {m.group(1)} {nxt}".strip()
                label = f"{part} / {sub}" if part else sub
            elif _GENERAL.match(line) and len(line) < 60 and not _TOC_ENTRY.search(line):
                sub = re.sub(r"\s+", " ", f"{line} {nxt}" if line.lower().startswith("when there is") else line)
                if sub.lower().startswith(_TOP_LEVEL):
                    part = ""
                label = f"{part} / {sub}" if part else sub
            if label:
                events[page_no].append((offset, label))
            offset += len(raw)
    return events


class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 800, chunk_overlap: int = 150,
                 model: Optional[SentenceTransformer] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # reuse an already-loaded model when the caller has one
        self.model = model or SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """Split each page into chunks and tag every chunk with the policy section(s) it belongs to."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            add_start_index=True,
        )
        by_source: Dict[str, List[Any]] = defaultdict(list)
        for d in documents:
            by_source[d.metadata.get("source", "")].append(d)

        chunks: List[Any] = []
        for source, docs in by_source.items():
            docs.sort(key=lambda d: d.metadata.get("page", 0))
            pages = [(i, d.page_content) for i, d in enumerate(docs)]
            events = _section_events(pages)

            # Chunk the whole document, not page by page: policy sentences often run across a page break
            # (e.g. a deductible exception that starts on one page and ends on the next).
            text, page_starts, all_events = "", [], []
            for i, d in enumerate(docs):
                page_starts.append(len(text))
                all_events += [(len(text) + off, lbl) for off, lbl in events.get(i, [])]
                text += d.page_content + "\n"
            base_meta = {k: v for k, v in docs[0].metadata.items() if k != "page"}

            for ch in splitter.split_documents([Document(page_content=text, metadata=base_meta)]):
                start = ch.metadata.get("start_index", 0)
                end = start + len(ch.page_content)
                page_idx = max(i for i, s in enumerate(page_starts) if s <= start)
                end_idx = max(i for i, s in enumerate(page_starts) if s < end)
                ch.metadata["page"] = docs[page_idx].metadata.get("page", page_idx)
                # chunks can run onto the next page; keep the range so citations point at the right pages
                ch.metadata["page_end"] = docs[end_idx].metadata.get("page", end_idx)
                at_start = [lbl for off, lbl in all_events if off <= start]
                inside = [lbl for off, lbl in all_events if start < off < end]
                labels = ([at_start[-1]] if at_start else []) + inside
                ch.metadata["section"] = " | ".join(dict.fromkeys(labels))
                chunks.append(ch)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} section-tagged chunks.")
        return chunks

    @staticmethod
    def embed_text(chunk: Any) -> str:
        """What gets embedded / keyword-indexed: section label + chunk text."""
        section = (getattr(chunk, "metadata", None) or {}).get("section")
        return f"{section}\n{chunk.page_content}" if section else chunk.page_content

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [self.embed_text(chunk) for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

# Example usage
if __name__ == "__main__":

    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
