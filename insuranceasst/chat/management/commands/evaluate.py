# chat/management/commands/evaluate.py
"""
Run the policy Q&A test set (eval/scenarios.json) through the same RAG pipeline the chat uses and score each turn.

    python manage.py evaluate --label baseline
    python manage.py evaluate --model mistral --only A1 M1

Writes results.md / results.json to <repo>/eval-runs/<date>_<label>/ (git-ignored).
"""
import json
import re
import time
from datetime import date
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from backend.search__ import RAGSearch

SCENARIOS = Path(settings.BASE_DIR) / "eval" / "scenarios.json"
RUNS_DIR = Path(settings.BASE_DIR).parent / "eval-runs"

QUESTION_MARKER = re.compile(r"to give you a precise answer", re.I)
# questions that ask the customer what the policy says (the assistant's job), not facts only they know
BAD_QUESTION = re.compile(
    r"what (is|are) the (conditions|terms|requirements)\b"
    r"|what (specific |type of )*coverages? (does|do) (the|my|this) policy\b"
    r"|what (specific |type of )*coverages? (may|might|would) (apply|cover)\b"
    r"|which coverages? (may|might|would|does|do) (cover|apply|provide)"
    r"|what conditions"
    r"|what actions do you need"
    r"|how (long|soon) (can|should|do|must) (i|you)\b"
    r"|(does|do|will|would) (your|my|the|this) [\w ]{0,40}(apply to|cover (the|this|glass|damage|towing|rental|it)\b)",
    re.I,
)
INSURER_VOICE = re.compile(r"(?<![\"'“])\bwe (will|must|may|have the right|can|do not|don't|won't)\b", re.I)


def cited_numbers(answer: str):
    nums = set()
    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        nums.update(int(n) for n in re.findall(r"\d+", group))
    return nums


def question_part(answer: str) -> str:
    m = QUESTION_MARKER.search(answer)
    if m:
        return answer[m.end():]
    return "\n".join(line for line in answer.splitlines() if "?" in line)


def grade(turn: dict, answer: str, sources: list) -> dict:
    low = answer.lower()
    pages = turn.get("pages") or []
    def pages_of(s):  # a passage can run across a page break
        start = s.get("page")
        end = s.get("page_end") or start
        return set(range(start, end + 1)) if isinstance(start, int) else set()
    retrieved = sorted(set().union(*[pages_of(s) for s in sources])) if sources else []
    cited = cited_numbers(answer)
    invalid = sorted(n for n in cited if n < 1 or n > len(sources))
    cited_pages = set().union(*[pages_of(sources[n - 1]) for n in cited if 1 <= n <= len(sources)]) if cited else set()
    qpart = question_part(answer)
    qlines = [line for line in qpart.splitlines() if "?" in line]

    fails = []
    if pages and not set(retrieved) & set(pages):
        fails.append(f"retrieval: none of pages {pages} retrieved (got {retrieved})")
    if pages and not cited_pages & set(pages):
        fails.append(f"citation: cited pages {sorted(p for p in cited_pages if p)} miss {pages}")
    if invalid:
        fails.append(f"invalid citation numbers {invalid}")
    for group in turn.get("include", []):
        if not any(p.lower() in low for p in group):
            fails.append(f"missing: one of {group}")
    for rx in turn.get("exclude", []):
        m = re.search(rx, answer, re.I)
        if m:
            fails.append(f"forbidden: '{m.group(0)}'")
    if turn.get("questions") == "required":
        if not qlines:
            fails.append("questions: none asked")
        elif turn.get("ask") and not any(k in qpart.lower() for k in turn["ask"]):
            fails.append(f"questions: none about {turn['ask']}")
    for rx in turn.get("ask_not", []):
        m = re.search(rx, qpart, re.I)
        if m:
            fails.append(f"re-asked: '{m.group(0)}'")
    bad = BAD_QUESTION.search(qpart)
    if bad:
        fails.append(f"bad question (asks what the policy says): '{bad.group(0)}'")

    return {
        "pass": not fails,
        "fails": fails,
        "retrieved_pages": retrieved,
        "cited_pages": sorted(p for p in cited_pages if p),
        "questions": len(qlines),
        "insurer_voice": bool(INSURER_VOICE.search(answer)),
    }


class Command(BaseCommand):
    help = "Score the RAG assistant against eval/scenarios.json"

    def add_arguments(self, parser):
        parser.add_argument("--model", default=getattr(settings, "LLM_MODEL", "qwen2.5:7b"))
        parser.add_argument("--label", default=None, help="run folder suffix (default: model name)")
        parser.add_argument("--only", nargs="*", default=None, help="scenario id prefixes, e.g. A1 M2")
        parser.add_argument("--hybrid", choices=["on", "off"], default=None, help="override settings.RAG_HYBRID")
        parser.add_argument("--expand", choices=["on", "off"], default=None, help="override settings.RAG_EXPAND_QUERY")
        parser.add_argument("--neighbors", choices=["on", "off"], default=None, help="override settings.RAG_NEIGHBORS")
        parser.add_argument("--route", choices=["on", "off"], default=None, help="override settings.RAG_ROUTE")
        parser.add_argument("--top-k", type=int, default=5)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--regrade", default=None,
                            help="run folder name under eval-runs/: re-score its saved answers without calling the LLM")

    def handle(self, *args, **opts):
        data = json.loads(SCENARIOS.read_text(encoding="utf-8"))
        scenarios = [s for s in data["scenarios"]
                     if not opts["only"] or any(s["id"].startswith(p) for p in opts["only"])]

        if opts["regrade"]:
            out_dir = RUNS_DIR / opts["regrade"]
            saved = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))["results"]
            turns = {(s["id"], i): t for s in scenarios for i, t in enumerate(s["turns"], 1)}
            results = [{**r, **grade(turns[(r["id"], r["turn"])], r["answer"], r["sources"])}
                       for r in saved if (r["id"], r["turn"]) in turns]
            model = saved[0].get("model", "?") if saved else "?"
            return self._write(out_dir, opts["regrade"], model, results)

        label = opts["label"] or opts["model"].replace(":", "-")
        out_dir = RUNS_DIR / f"{date.today().isoformat()}_{label}"
        out_dir.mkdir(parents=True, exist_ok=True)

        rag = RAGSearch(
            persist_dir=settings.FAISS_DIR,
            backend=getattr(settings, "LLM_BACKEND", "ollama"),
            model_name=opts["model"],
            data_dir=settings.MEDIA_ROOT,
            hybrid=(opts["hybrid"] == "on") if opts["hybrid"] else getattr(settings, "RAG_HYBRID", False),
            expand_query=(opts["expand"] == "on") if opts["expand"] else getattr(settings, "RAG_EXPAND_QUERY", False),
            neighbors=(opts["neighbors"] == "on") if opts["neighbors"] else getattr(settings, "RAG_NEIGHBORS", False),
            route=(opts["route"] == "on") if opts["route"] else getattr(settings, "RAG_ROUTE", False),
        )
        rag.llm.seed = opts["seed"]
        self.stdout.write(f"seed={opts['seed']} hybrid={rag.hybrid} expand_query={rag.expand_query} neighbors={rag.neighbors} route={rag.route} top_k={opts['top_k']}")

        results = []
        for sc in scenarios:
            history = []
            for i, turn in enumerate(sc["turns"], 1):
                t0 = time.time()
                hits, messages = rag.prepare(turn["user"], doc=sc["policy"], history=history,
                                             facts=sc.get("facts"), top_k=opts["top_k"])
                answer = rag.llm.chat(messages, max_new_tokens=700) if hits else "No relevant passages found."
                sources = rag.sources_from(hits)
                g = grade(turn, answer, sources)
                results.append({"id": sc["id"], "turn": i, "policy": sc["policy"], "model": opts["model"], "user": turn["user"],
                                "truth": turn.get("truth", ""), "answer": answer, "sources": sources,
                                "seconds": round(time.time() - t0, 1), **g})
                self.stdout.write(f"  {sc['id']} turn {i}: {'pass' if g['pass'] else 'fail'} ({time.time() - t0:.0f}s)")
                history += [{"role": "user", "content": turn["user"]}, {"role": "assistant", "content": answer}]

        self._write(out_dir, label, opts["model"], results)

    def _write(self, out_dir, label, model, results):
        passed = sum(r["pass"] for r in results)
        retr = [r for r in results if r["truth"] and not any(f.startswith("retrieval") for f in r["fails"])]
        summary = (f"{passed}/{len(results)} turns pass · retrieval found an expected page in "
                   f"{len(retr)}/{len(results)} · model {model}")
        for r in results:
            if not r["pass"]:
                self.stdout.write(f"FAIL {r['id']} turn {r['turn']}  " + "; ".join(r["fails"]))
        self.stdout.write(self.style.SUCCESS(summary))

        (out_dir / "results.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
        lines = [f"# Eval run: {label}", "", f"**{summary}**", "",
                 "| Scenario | Turn | Result | Retrieved pages | Cited | Problems |", "|---|---|---|---|---|---|"]
        for r in results:
            lines.append(f"| {r['id']} | {r['turn']} | {'✅' if r['pass'] else '❌'} | {r['retrieved_pages']} | "
                         f"{r['cited_pages']} | {'<br>'.join(r['fails']) or '–'} |")
        for r in results:
            lines += ["", f"## {r['id']} · turn {r['turn']} {'✅' if r['pass'] else '❌'} ({r['seconds']}s)", "",
                      f"**Customer:** {r['user']}", "", f"**Correct answer:** {r['truth']}", "",
                      "**Sources:** " + ", ".join(f"[{n}] p.{s['page']}" for n, s in enumerate(r["sources"], 1)), "",
                      "**Assistant:**", "", r["answer"]]
        (out_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")
        self.stdout.write(f"Wrote {out_dir}")
