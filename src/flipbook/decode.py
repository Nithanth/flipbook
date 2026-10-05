"""Decode stored token_ids back into text for inspection paths.

A sample's `text` is only the post-thinking message the renderer parsed;
`token_ids` carries the whole generation. The CLI `show` and the API row
endpoint both need the part the renderer dropped.
"""

from functools import lru_cache

from flipbook.graders import strip_control_tokens


@lru_cache(maxsize=4)
def tokenizer(model_id: str):
    # per-id decode is what we need (token boundaries), so keep the raw tokenizer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return get_tokenizer(model_id)


def thinking(model_id: str | None, token_ids: list | None, text: str | None) -> str | None:
    """Decoded generation preceding the stored final message, or None."""
    if not model_id or not token_ids or not text:
        return None
    full = tokenizer(model_id).decode([int(i) for i in token_ids])
    # WHY a 40-char prefix: the stored text may be whitespace-trimmed relative
    # to the decode, so an exact full-string find can miss
    offset = full.find(text[:40])
    if offset < 0:
        return None
    return strip_control_tokens(full[:offset]).strip()
