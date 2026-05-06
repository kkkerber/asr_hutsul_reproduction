"""
utils/text_normalization.py
===========================

Reusable text-normalization utilities for the Hutsul ASR pipeline.

The paper specifies the following transcript pre-processing:

* lowercase the whole transcription,
* strip punctuation,
* collapse repeated whitespace,
* preserve the 33 Ukrainian letters (а … я) plus apostrophe,
* unify the three apostrophe variants ('ʼ', '’', "'") to U+0027,
* unicode-normalise to NFC.

This module exposes both a function-level API (``normalize_text``) and
a configurable :class:`TextNormalizer` so that callers can opt out of
individual steps (e.g. keep digits when normalising spoken-numeral
references).
"""

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Apostrophe-like characters that should all collapse to ``"'"``.
APOSTROPHE_VARIANTS: Final[str] = "ʼ’`´‘ʹ"

# Punctuation we always strip.  We list them explicitly rather than
# using ``string.punctuation`` so that we have full control over which
# characters survive (apostrophe must not be deleted).
PUNCTUATION_CHARS: Final[str] = (
    r".,!?;:()[]{}<>«»\"„“”/\\|—–-…•·"
    r"§¶†‡@#$%^&*+=~_"
)

# Compiled regular expressions (compiled once at import time).
_RE_MULTI_WS = re.compile(r"\s+")
_RE_DIGITS = re.compile(r"\d+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class TextNormalizer:
    """Configurable text normaliser.

    Parameters
    ----------
    lowercase
        Lowercase the input.  Default ``True``.
    nfc
        Apply Unicode NFC normalisation before any other step.  Default
        ``True``.
    unify_apostrophes
        Map ``ʼ``/``’``/etc. to ``"'"``.  Default ``True``.
    strip_punctuation
        Remove the characters listed in :data:`PUNCTUATION_CHARS`.
        Default ``True``.
    remove_digits
        Drop digits.  The Hutsul corpus contains very few digits but
        when they do appear they are typically read aloud as words and
        so we strip them by default.  Default ``True``.
    keep_chars
        Whitelist of characters to keep.  Anything outside this set
        (and not handled by another step) is dropped.  Default: the
        Ukrainian alphabet + apostrophe + space.
    collapse_whitespace
        Replace any whitespace run with a single space.  Default
        ``True``.
    strip
        Strip leading/trailing whitespace.  Default ``True``.
    """

    lowercase: bool = True
    nfc: bool = True
    unify_apostrophes: bool = True
    strip_punctuation: bool = True
    remove_digits: bool = True
    keep_chars: Optional[Set[str]] = None
    collapse_whitespace: bool = True
    strip: bool = True

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.keep_chars is None:
            # Default whitelist.  We instantiate from the project-level
            # constant so that callers who modify the alphabet only have
            # to do it in ``config.py``.
            self.keep_chars = set(ALLOWED_CHARS)

        # Pre-compute punctuation translation table (str.translate is
        # significantly faster than running a regex over every sample).
        if self.strip_punctuation:
            self._punct_table = str.maketrans(
                {ch: " " for ch in PUNCTUATION_CHARS}
            )
        else:
            self._punct_table = None  # type: ignore[assignment]

        if self.unify_apostrophes:
            self._apostrophe_table = str.maketrans(
                {ch: APOSTROPHE for ch in APOSTROPHE_VARIANTS}
            )
        else:
            self._apostrophe_table = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    def __call__(self, text: str) -> str:
        return self.normalize(text)

    # ------------------------------------------------------------------
    def normalize(self, text: str) -> str:
        """Normalise a single transcription string."""
        if text is None:
            return ""

        if not isinstance(text, str):
            # Datasets can hand us bytes for some columns — coerce.
            try:
                text = str(text)
            except Exception as exc:  # pragma: no cover — defensive
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

        # Whitelist-filter any remaining non-allowed characters.
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

    # ------------------------------------------------------------------
    def normalize_batch(self, texts: Iterable[str]) -> List[str]:
        """Normalise an iterable of transcriptions."""
        return [self.normalize(t) for t in texts]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


_DEFAULT_NORMALIZER: Optional[TextNormalizer] = None


def build_default_normalizer() -> TextNormalizer:
    """Return (and cache) the default project-wide :class:`TextNormalizer`."""
    global _DEFAULT_NORMALIZER
    if _DEFAULT_NORMALIZER is None:
        _DEFAULT_NORMALIZER = TextNormalizer()
    return _DEFAULT_NORMALIZER


def normalize_text(text: str) -> str:
    """Normalise ``text`` with the project-default normaliser."""
    return build_default_normalizer().normalize(text)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def _smoke_test() -> None:  # pragma: no cover
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

    # Sanity: every default-allowed character must come from the
    # Ukrainian alphabet, apostrophe, or whitespace.
    for ch in ALLOWED_CHARS:
        assert ch == APOSTROPHE or ch == " " or ch in UKRAINIAN_ALPHABET


if __name__ == "__main__":  # pragma: no cover
    _smoke_test()
