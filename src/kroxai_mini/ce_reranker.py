from __future__ import annotations

from typing import Any, List, Optional, Tuple


class CrossEncoderReranker:
    """
    Minimal cross-encoder reranker wrapper.

    - Tries to import sentence_transformers.CrossEncoder lazily
    - If unavailable or model load fails, .ok will be False and rerank() is a no-op (returns top-k input order)
    - Accepts hits as list of (doc, score) where doc may be any object (dicts with 'text' are supported)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._ok = False
        self._ce = None
        try:
            import sentence_transformers as st  # type: ignore
            self._ce = st.CrossEncoder(model_name, device=device)
            self._ok = True
        except Exception:
            self._ok = False
            self._ce = None

    @property
    def ok(self) -> bool:
        return bool(self._ok and self._ce is not None)

    def info(self) -> dict:
        return {
            "ok": self.ok,
            "model": self.model_name if self.ok else None,
            "device": getattr(self._ce, "device", None) if self.ok else None,
        }

    def rerank(self, query: str, hits: List[Tuple[Any, float]], k: int = 5) -> List[Tuple[Any, float]]:
        if not self.ok or not hits:
            return hits[:k]
        pairs = []
        docs: List[Any] = []
        for doc, _ in hits:
            # Support dicts with 'text', otherwise stringify
            if isinstance(doc, dict):
                txt = str(doc.get("text", ""))
            else:
                txt = str(getattr(doc, "text", "") or str(doc))
            pairs.append([query, txt])
            docs.append(doc)
        try:
            scores = self._ce.predict(pairs)  # type: ignore[attr-defined]
        except Exception:
            return hits[:k]
        out = [(docs[i], float(scores[i])) for i in range(len(docs))]
        out.sort(key=lambda t: t[1], reverse=True)
        return out[:k]


_ce_instance: Optional[CrossEncoderReranker] = None


def get_ce_reranker(model_name: Optional[str] = None, device: Optional[str] = None) -> CrossEncoderReranker:
    global _ce_instance
    if _ce_instance is None:
        _ce_instance = CrossEncoderReranker(model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    return _ce_instance
