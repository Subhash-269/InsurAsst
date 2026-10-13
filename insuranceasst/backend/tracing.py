"""
MLflow helpers: request tracing and experiment setup, all optional.

Tracing records each chat turn as a tree of spans (routing -> retrieval -> LLM) with inputs, outputs and
timings, viewable in `mlflow ui`. If mlflow isn't installed or MLFLOW_TRACING=0, every helper here is a
no-op and the app behaves exactly as before.
"""
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

try:
    import mlflow
    from mlflow.entities import SpanType
    if not hasattr(mlflow, "start_span"):  # MLflow < 2.14 has tracking but no tracing
        raise ImportError("mlflow too old for tracing")
except ImportError:  # tracing is optional
    mlflow = None
    SpanType = None

_configured = False


def enabled() -> bool:
    return mlflow is not None and os.getenv("MLFLOW_TRACING", "1") != "0"


def configure(tracking_uri: str, experiment: str, artifact_root: Optional[Path] = None) -> Optional[str]:
    """Point MLflow at the local store and select (creating if needed) an experiment. Returns its id."""
    global _configured
    if mlflow is None:
        return None
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        # keep artifacts next to the database instead of wherever the process happens to start
        location = (artifact_root / experiment).resolve().as_uri() if artifact_root else None
        exp_id = mlflow.create_experiment(experiment, artifact_location=location)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    _configured = True
    return exp_id


def trace(name: Optional[str] = None, span_type: Optional[str] = None):
    """@trace(...) -> mlflow.trace when tracing is on, otherwise the function unchanged.
    Works for plain functions and generators (streamed answers)."""
    def wrap(fn):
        if not enabled():
            return fn
        kwargs = {"name": name or fn.__name__}
        if span_type:
            kwargs["span_type"] = span_type
        return mlflow.trace(**kwargs)(fn)
    return wrap


@contextmanager
def span(name: str, span_type: Optional[str] = None, **attributes):
    """`with span("step") as s:` - a child span (or a dummy object when tracing is off)."""
    if not enabled():
        yield _NoSpan()
        return
    kwargs = {"name": name}
    if span_type:
        kwargs["span_type"] = span_type
    with mlflow.start_span(**kwargs) as s:
        if attributes:
            s.set_attributes(attributes)
        yield s


class TurnTrace:
    """One trace per chat turn, kept open across Django's streamed response.

    The view runs retrieval, returns, and Django then iterates the answer generator, so a normal
    `with` block would close the trace before the answer is written. This opens the root span in
    the view and closes it when the stream ends (or the client disconnects)."""

    def __init__(self, name: str, inputs: dict):
        self._cm = None
        self.span = _NoSpan()
        if enabled():
            self._cm = mlflow.start_span(name=name, span_type=SpanType.CHAIN)
            self.span = self._cm.__enter__()
            self.span.set_inputs(inputs)

    def finish(self, outputs=None, error: Optional[BaseException] = None):
        if self._cm is None:
            return
        try:
            if outputs is not None:
                self.span.set_outputs(outputs)
            if error is not None:
                self._cm.__exit__(type(error), error, error.__traceback__)
            else:
                self._cm.__exit__(None, None, None)
        finally:
            self._cm = None


class _NoSpan:
    def set_inputs(self, *a, **k): pass
    def set_outputs(self, *a, **k): pass
    def set_attributes(self, *a, **k): pass
    def set_attribute(self, *a, **k): pass
