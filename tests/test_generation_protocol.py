"""Independent UTF-8 hashes pin the exact generation prompt bytes."""

import hashlib

import pytest

from parallel_search_evals.config import (
    CLOSED_BOOK_INSTRUCTIONS,
    FINAL_ANSWER_INSTRUCTIONS,
    RESEARCH_INSTRUCTIONS,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        (CLOSED_BOOK_INSTRUCTIONS, "7aa27ada960d3bc183fae55503d008b3e0d69eeb29f989687c11f624f66dacb7"),
        (RESEARCH_INSTRUCTIONS, "def8b37b8b669d1adb91e61d52677ab6ac6889fbe90d22e53e8a6e6815527d1c"),
        (FINAL_ANSWER_INSTRUCTIONS, "d00d3a3da922844966da99bfa81cb94f9d25cca2536aa1354138784f48bd949a"),
    ],
)
def test_pinned_generation_prompt_bytes(text, expected):
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == expected
