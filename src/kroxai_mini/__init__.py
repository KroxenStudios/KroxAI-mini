"""kroxai_mini package public surface.

Keep imports lightweight so importing ``kroxai_mini`` doesn't pull in heavy
optional dependencies (numpy/torch/transformers).

Expose light components (for example ``SimpleTokenizer``) immediately.
Expose heavy components like ``KroxAI``, ``TransformerLM``, and ``data_utils`` lazily
when first accessed.
"""

from .tokenizer import SimpleTokenizer

__all__ = ["SimpleTokenizer", "data_utils", "KroxAI", "TransformerLM"]

__version__ = "0.0.1"


def _load_kroxai_class():
    """Lazy import of the KroxAI implementation.

    This may raise ImportError if optional runtime dependencies are
    missing (for example torch/transformers). Callers that need the
    full implementation should handle that.
    """
    from .torch_chat import KroxAI

    return KroxAI


def _load_transformer():
    """Lazy import of TransformerLM.

    This requires numpy.
    """
    from .transformer import TransformerLM

    return TransformerLM


def _load_data_utils():
    """Lazy import of data utilities.

    This requires numpy.
    """
    from . import data as data_utils

    return data_utils


def __getattr__(name: str):
    # PEP 562 module-level lazy attribute access
    if name == "KroxAI":
        return _load_kroxai_class()
    if name == "TransformerLM":
        return _load_transformer()
    if name == "data_utils":
        return _load_data_utils()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + ["KroxAI", "TransformerLM", "data_utils"])
