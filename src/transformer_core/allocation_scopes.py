"""Opt-in scalar-only profiler ranges; default paths do not import the profiler."""

import json
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

_ENABLED = ContextVar("transformer_allocation_scopes", default=False)
PREFIX = "transformer_allocation:"


@contextmanager
def allocation_scopes():
    """Enable ranges in this context, restoring the prior state even after failure."""
    token = _ENABLED.set(True)
    try:
        yield
    finally:
        _ENABLED.reset(token)


def scope(site, **metadata):
    if not _ENABLED.get():
        return nullcontext()
    if not isinstance(site, str) or not site:
        raise ValueError("site must be a nonempty string")
    if any(type(v) not in (str, int, bool) for v in metadata.values()):
        raise ValueError("range metadata must be scalars, never tensor references")
    from torch.profiler import record_function

    return record_function(PREFIX + json.dumps(dict(site=site, **metadata), sort_keys=True))
