import json, os
from pathlib import Path
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from backend.search__ import RAGSearch
from django.http import StreamingHttpResponse
import uuid, cv2, numpy as np
from ultralytics import YOLO


# ---------- RAG singleton ----------
_rag = None
def _get_rag():
    global _rag
    if _rag is None:
        _rag = RAGSearch(
            persist_dir=getattr(settings, "FAISS_DIR", "faiss_store"),
            backend=getattr(settings, "LLM_BACKEND", "ollama"),
            model_name=getattr(settings, "LLM_MODEL", "mistral"),
        )
    return _rag

# ---------- Pages ----------
def index(request):
    # Dark theme is purely front-end (CSS)
    return render(request, "chat/index.html")

def health(request):
    return JsonResponse({"ok": True})

# NEW: list indexed docs (from FAISS metadata)
def indexed_docs(request):
    if request.method != "GET":
        return JsonResponse({"error": "GET only"}, status=405)
    try:
        rag = _get_rag()
        items = rag.vectorstore.list_indexed_sources()
        # {source, name, count}
        return JsonResponse({"docs": items})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def _list_indexed_docs():
    # Return just the file basenames to match the filter
    base = Path(getattr(settings, "MEDIA_ROOT", settings.BASE_DIR / "data"))
    out = []
    for root, _, files in os.walk(base):
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in {".pdf", ".txt", ".docx", ".csv", ".xlsx", ".json"}:
                out.append(f)  # basename
    out.sort()
    return out

def index(request):
    # Pass document options to the template
    return render(request, "chat/index.html", {"docs": _list_indexed_docs()})

# ---------- Chat API ----------
@csrf_exempt
def chat_api(request):
    if request.method == "GET":
        q = request.GET.get("q")
        doc = request.GET.get("doc") or None
        if q:
            return JsonResponse({"answer": _get_rag().search_and_summarize(q, top_k=8, doc=doc)})
        return JsonResponse({"hint": "POST JSON {message, doc?} or GET /api/chat/?q=...&doc=<filename>"})

    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8"))
        msg = (payload.get("message") or "").strip()
        doc = (payload.get("doc") or "").strip() or None
        if not msg:
            return JsonResponse({"error": "Empty message"}, status=400)
        answer = _get_rag().search_and_summarize(msg, top_k=8, doc=doc)
        return JsonResponse({"answer": answer})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# ---------- Files & Indexing ----------
def _data_dir() -> Path:
    return Path(getattr(settings, "MEDIA_ROOT", settings.BASE_DIR / "data"))

def _public_url(rel_path: str) -> str:
    # /media/<filename>
    return f"{settings.MEDIA_URL}{rel_path}".replace("\\", "/")

def _allowed(filename: str) -> bool:
    allowed_ext = {".pdf", ".txt", ".docx", ".csv", ".xlsx", ".json"}
    return Path(filename).suffix.lower() in allowed_ext

def list_files(request):
    """GET JSON list of files available for preview (from ./data)."""
    if request.method != "GET":
        return JsonResponse({"error": "GET only"}, status=405)
    base = _data_dir()
    base.mkdir(parents=True, exist_ok=True)
    items = []
    for root, _, files in os.walk(base):
        for f in files:
            if not _allowed(f): 
                continue
            full = Path(root) / f
            rel = full.relative_to(base).as_posix()
            items.append({
                "name": f,
                "rel": rel,
                "size": full.stat().st_size,
                "url": _public_url(rel),
                "ext": Path(f).suffix.lower()
            })
    items.sort(key=lambda x: x["name"].lower())
    return JsonResponse({"files": items})

@csrf_exempt
def upload_file(request):
    """POST a file; save to ./data; optionally reindex via separate call."""
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    if "file" not in request.FILES:
        return JsonResponse({"error": "No file field named 'file' in form-data"}, status=400)

    file = request.FILES["file"]
    if not _allowed(file.name):
        return JsonResponse({"error": "Unsupported file type"}, status=400)

    storage = FileSystemStorage(location=_data_dir())
    filename = storage.save(file.name, file)
    return JsonResponse({"ok": True, "saved_as": filename, "url": _public_url(filename)})

@csrf_exempt
def reindex(request):
    """POST: rebuild FAISS index from current ./data. (Blocking)"""
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    try:
        # Build index synchronously using your vector store (same as build_index cmd)
        from backend.data_loader import load_all_documents
        from backend.vectorstore import FaissVectorStore

        docs = load_all_documents(str(_data_dir()))
        store = FaissVectorStore(getattr(settings, "FAISS_DIR", "faiss_store"))
        store.build_from_documents(docs)
        if hasattr(store, "save"):
            store.save()
        return JsonResponse({"ok": True, "message": "FAISS index rebuilt.", "doc_count": len(docs)})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)

@csrf_exempt
def chat_stream(request):
    """
    POST { "message": "...", "doc": "Allstate.pdf" }
    Streams back the answer in chunks.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    try:
        payload = json.loads(request.body.decode("utf-8"))
        msg = (payload.get("message") or "").strip()
        doc = (payload.get("doc") or "").strip() or None
        if not msg:
            return JsonResponse({"error": "Empty message"}, status=400)

        rag = _get_rag()

        # Build retrieval context (mirror of search_and_summarize)
        results = rag.vectorstore.query(msg, top_k=8, allowed_sources=[doc] if doc else None)
        texts = [r.get("metadata", {}).get("text", "") for r in results if r.get("metadata")]
        uniq, seen = [], set()
        for t in texts:
            t2 = t.strip()
            if t2 and t2 not in seen:
                seen.add(t2)
                uniq.append(t2)
        context = "\n\n---\n\n".join(uniq)[:8000]
        if not context:
            # simple immediate response
            return StreamingHttpResponse(
                iter(["No matches found in the selected policy. Try Rebuild Index or choose another document."]),
                content_type="text/plain",
            )
        prompt = rag._build_prompt(msg, context)  # uses the helper already in your class

        def gen():
            backend = getattr(settings, "LLM_BACKEND", "ollama").lower()
            if backend == "ollama":
                # True streaming from Ollama
                import ollama
                model = getattr(rag.llm, "model", getattr(settings, "LLM_MODEL", "mistral"))
                for part in ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    options={"temperature": 0.2, "top_p": 0.95},
                ):
                    # parts have shape: {"message": {"content": "..."}, ...}
                    chunk = (part.get("message") or {}).get("content") or ""
                    if chunk:
                        yield chunk
            else:
                # Fallback: generate full text, then drip it out as "fake" streaming
                text = rag.llm.generate(prompt, max_new_tokens=256)
                for piece in text.split():
                    yield piece + " "
            # done
        return StreamingHttpResponse(gen(), content_type="text/plain")
    except Exception as e:
        return StreamingHttpResponse(iter([f"Error: {e}"]), content_type="text/plain", status=500)

# lazy globals
_yolo = None
_CARDD_CLASSES = ["crack", "dent", "glass shatter", "lamp broken", "scratch", "tire flat"]

def _get_yolo():
    global _yolo
    if _yolo is None:
        model_path = getattr(settings, 'CARDD_MODEL_PATH', 'yolov11-seg-cardd.pt')
        _yolo = YOLO(model_path)
    return _yolo

def _media_root() -> Path:
    return Path(getattr(settings, "MEDIA_ROOT", settings.BASE_DIR / "data"))

def _media_url(rel: str) -> str:
    return f"{settings.MEDIA_URL}{rel}".replace("\\", "/")

@csrf_exempt
def vision_analyze(request):
    """
    POST multipart/form-data with 'image'
    Returns: { ok, annotated_url, counts:{}, detections:[...], summary }
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    if "image" not in request.FILES:
        return JsonResponse({"error": "No 'image' file found"}, status=400)

    try:
        img_file = request.FILES["image"]
        # save original
        claims_dir = _media_root() / "images" / "claims"
        claims_dir.mkdir(parents=True, exist_ok=True)
        stem = uuid.uuid4().hex
        ext = Path(img_file.name).suffix.lower() or ".jpg"
        raw_path = claims_dir / f"{stem}{ext}"
        with open(raw_path, "wb") as f:
            for chunk in img_file.chunks():
                f.write(chunk)

        # run YOLO seg
        model = _get_yolo()
        results = model.predict(source=str(raw_path), imgsz=640, conf=0.25, device=0)
        res = results[0]

        # annotated image (with masks/boxes/labels)
        im_annotated = res.plot()  # numpy BGR
        ann_path = claims_dir / f"{stem}_annotated.jpg"
        cv2.imwrite(str(ann_path), im_annotated)

        # detections + counts
        detections = []
        counts = {name: 0 for name in _CARDD_CLASSES}
        for i, b in enumerate(res.boxes):
            cls_id = int(b.cls[0])
            score = float(b.conf[0])
            xyxy  = [float(v) for v in b.xyxy[0].tolist()]
            name  = _CARDD_CLASSES[cls_id] if 0 <= cls_id < len(_CARDD_CLASSES) else str(cls_id)
            counts[name] = counts.get(name, 0) + 1
            det = {"class_id": cls_id, "class_name": name, "score": score, "box": xyxy}
            # polygons if available
            if res.masks is not None and i < len(res.masks.xy):
                poly = res.masks.xy[i]
                if isinstance(poly, list):
                    det["polygons"] = [p.tolist() for p in poly]
                else:
                    det["polygons"] = [poly.tolist()]
            detections.append(det)

        # build a concise fact payload for LLM summary
        facts = {
            "counts": {k: v for k, v in counts.items() if v > 0},
            "total_instances": sum(counts.values()),
        }
        fact_str = json.dumps(facts, indent=2)

        # readable summary with your existing LLM (Mistral)
        rag = _get_rag()
        summary_prompt = (
            "You are an auto-claims assistant. Given these vision detections, write a short, plain-English summary "
            "of the observed damage (1-3 sentences), mentioning the types and approximate counts. Be neutral and factual.\n\n"
            f"Detections JSON:\n{fact_str}\n\nSummary:"
        )
        summary = rag.llm.generate(summary_prompt, max_new_tokens=120).strip()

        rel_ann = ann_path.relative_to(_media_root()).as_posix()
        return JsonResponse({
            "ok": True,
            "annotated_url": _media_url(rel_ann),
            "counts": facts["counts"],
            "detections": detections,
            "summary": summary
        })
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)