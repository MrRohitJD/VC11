"""Request-local, public execution events; never record model reasoning."""
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

_activity = ContextVar("mailbox_activity", default=None)


class SearchActivity:
    def __init__(self, callback=None):
        self.steps = []
        self.callback = callback
        self.started = perf_counter()

    def emit(self, action, observation, *, status="completed", **details):
        event = {"action": action, "observation": observation, "status": status,
                 "elapsed_ms": round((perf_counter() - self.started) * 1000), **details}
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
