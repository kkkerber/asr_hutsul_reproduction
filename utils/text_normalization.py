from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Final, Iterable, List, Optional, Set

from config import (
    ALLOWED_CHARS,
    APOSTROPHE,
    UKRAINIAN_ALPHABET,
)

logger = logging.getLogger(__name__)

APOSTROPHE_VARIANTS: Final[str] = "ʼ’`´‘ʹ"

PUNCTUATION_CHARS: Final[str] = (
    r".,!?;:()[]{}<>«»\"„“”/\\|—–-…•·"
    r"§¶†‡@#$%^&*+=~_"
)

_RE_MULTI_WS = re.compile(r"\s+")
_RE_DIGITS = re.compile(r"\d+")


@dataclass
class TextNormalizer:

    lowercase: bool = True
    nfc: bool = True
    unify_apostrophes: bool = True
    strip_punctuation: bool = True
    remove_digits: bool = True
    keep_chars: Optional[Set[str]] = None
    collapse_whitespace: bool = True
    strip: bool = True

    def __post_init__(self) -> None:
        if self.keep_chars is None:
            self.keep_chars = set(ALLOWED_CHARS)

        if self.strip_punctuation:
            self._punct_table = str.maketrans(
                {ch: " " for ch in PUNCTUATION_CHARS}
            )
        else:
            self._punct_table = None

        if self.unify_apostrophes:
            self._apostrophe_table = str.maketrans(
                {ch: APOSTROPHE for ch in APOSTROPHE_VARIANTS}
            )
        else:
            self._apostrophe_table = None

    def __call__(self, text: str) -> str:
        return self.normalize(text)

    def normalize(self, text: str) -> str:
        if text is None:
            return ""

        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception as exc:
                logger.warning("Could not coerce %r to str: %s", text, exc)
                return ""

        if self.nfc:
            text = unicodedata.normalize("NFC", text)

        if self.unify_apostrophes and self._apostrophe_table is not None:
            text = text.translate(self._apostrophe_table)

        if self.lowercase:
            text = text.lower()

        if self.strip_punctuation and self._punct_table is not None:
            text = text.translate(self._punct_table)

        if self.remove_digits:
            text = _RE_DIGITS.sub(" ", text)

        if self.keep_chars is not None:
            text = "".join(
                ch if (ch in self.keep_chars or ch.isspace()) else " "
                for ch in text
            )

        if self.collapse_whitespace:
            text = _RE_MULTI_WS.sub(" ", text)

        if self.strip:
            text = text.strip()

        return text

    def normalize_batch(self, texts: Iterable[str]) -> List[str]:
        return [self.normalize(t) for t in texts]


_DEFAULT_NORMALIZER: Optional[TextNormalizer] = None


def build_default_normalizer() -> TextNormalizer:

    global _DEFAULT_NORMALIZER
    if _DEFAULT_NORMALIZER is None:
        _DEFAULT_NORMALIZER = TextNormalizer()
    return _DEFAULT_NORMALIZER


def normalize_text(text: str) -> str:
    return build_default_normalizer().normalize(text)


def _smoke_test() -> None:
    samples = [
        "Привіт, світе!  ",
        "Він   сказав: «Не йди!»",
        "Це дзвонить о 8:30 ранку.",
        "Дев’ять — найкраще число!",
        "ТЕСТ З АПОСТРОФОМ ʼ та ’",
    ]
    norm = build_default_normalizer()
    for s in samples:
        print(f"{s!r:60s} -> {norm(s)!r}")

    for ch in ALLOWED_CHARS:
        assert ch == APOSTROPHE or ch == " " or ch in UKRAINIAN_ALPHABET


if __name__ == "__main__":
    _smoke_test()
