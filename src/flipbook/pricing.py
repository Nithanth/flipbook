"""Per-1M-token price table backing forecast and budget math.
`cached` is the reduced prefill rate for cached prefixes. From TML docs: https://tinker-docs.thinkingmachines.ai/tinker/models/
"""

PRICES: dict[str, dict[str, dict[str, float]]] = {
    "thinkingmachines/Inkling-Small": {
        "prefill": {"list": 1.16, "discount": 0.58, "cached": 0.116},
        "sample": {"list": 2.88, "discount": 1.44},
        "train": {"list": 3.46, "discount": 1.73},
    },
    "thinkingmachines/Inkling": {
        "prefill": {"list": 3.74, "discount": 1.87},
        "sample": {"list": 9.36, "discount": 4.68},
        "train": {"list": 11.22, "discount": 5.61},
    },
}

STORAGE_USD_PER_GB_MONTH = 0.10


def estimate_usd(model: str, op: str, tokens: int, *, list_price: bool = False) -> float:
    tier = "list" if list_price else "discount"
    return PRICES[model][op][tier] * tokens / 1e6
