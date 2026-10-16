"""Helpers shared by the rule modules. Not a rule itself, so it registers nothing."""

from __future__ import annotations

import ast

from .. import models
from ..core import CallSite

# Request methods of the Anthropic SDK that send a generation request. ``messages.count_tokens``
# is left out: it takes no sampling or max_tokens, and what it rejects isn't documented.
ANTHROPIC_REQUEST_CALLS = ("messages.create", "messages.stream", "messages.parse")

SAMPLING_PARAMS = ("temperature", "top_p", "top_k")

# Values that tell the SDK to leave a field out of the request.
_UNSET_NAMES = frozenset({"NOT_GIVEN", "NotGiven", "omit", "Omit"})


def site_traits(site: CallSite) -> models.ModelTraits | None:
    """Traits of the call's model, or None when the model can't be resolved or isn't in the table."""
    return models.traits(site.model())


def dict_field(node: ast.expr | None, key: str) -> ast.expr | None:
    """The value node for a constant string ``key`` in a dict display, else None."""
    if not isinstance(node, ast.Dict):
        return None
    for k, v in zip(node.keys, node.values):
        if isinstance(k, ast.Constant) and k.value == key:
            return v
    return None


def dict_str(node: ast.expr | None, key: str) -> str | None:
    value = dict_field(node, key)
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def has_splat(site: CallSite) -> bool:
    """True when the call also passes ``**kwargs``, which may add or override fields we can't see."""
    return any(k.arg is None for k in site.node.keywords)


def request_fields(site: CallSite, name: str) -> list[ast.expr]:
    """Every node that sends body field ``name``: the keyword argument and a key in a literal ``extra_body``."""
    found = []
    if name in site.kwargs:
        found.append(site.kwargs[name])
    in_extra = dict_field(site.kwargs.get("extra_body"), name)
    if in_extra is not None:
        found.append(in_extra)
    return found


def may_be_unset(node: ast.expr) -> bool:
    """True when the expression is or may evaluate to None / NOT_GIVEN / omit, i.e. the field may not be sent."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and sub.value is None:
            return True
        if isinstance(sub, ast.Name) and sub.id in _UNSET_NAMES:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in _UNSET_NAMES:
            return True
    return False
