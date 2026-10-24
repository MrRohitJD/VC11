"""Request-local, public execution events; never record model reasoning."""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from time import perf_counter

_activity = ContextVar("mailbox_activity", default=None)


class SearchActivity:
    def __init__(self, callback=None):
        self.steps = []
        self.callback = callback
        self.started = perf_counter()
        self._lock = Lock()
        self._sequence = 0

    def emit(self, action, observation, *, status="completed", **details):
        # Events are intentionally an execution summary, not a trace of model
        # prompts, hidden reasoning, tool arguments, or email bodies.  Keep
        # only UI-safe routing/progress metadata supplied by backend stages.
        public_keys = {
            "count", "evaluated", "source_count", "intent", "strategy", "providers",
            "provider", "parallel", "branch", "branch_count", "plan", "routing_reason",
            "duration_ms", "lane", "parallel_group",
        }
        safe_details = {
            key: value for key, value in details.items()
            if key in public_keys and value is not None
        }
        with self._lock:
            self._sequence += 1
            event = {
                "event_id": self._sequence,
                "phase": action,
                "action": action,
                "observation": observation,
                "status": status,
                "elapsed_ms": round((perf_counter() - self.started) * 1000),
                **safe_details,
            }
            self.steps.append(event)
        if self.callback:
            self.callback(event)
        return event


@contextmanager
def activity_context(callback=None):
    existing = _activity.get()
    if existing is not None and callback is None:
        yield existing
        return
    recorder = SearchActivity(callback)
    token = _activity.set(recorder)
    try:
        yield recorder
    finally:
        _activity.reset(token)


def emit_activity(action, observation, **details):
    recorder = _activity.get()
    if recorder:
        return recorder.emit(action, observation, **details)
