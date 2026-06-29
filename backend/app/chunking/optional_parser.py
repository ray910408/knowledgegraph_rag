from __future__ import annotations

from typing import Any, Protocol


class OptionalStructuredParser(Protocol):
    def parse(self, text: str, *, language_hint: str | None = None) -> dict[str, Any] | None:
        ...
