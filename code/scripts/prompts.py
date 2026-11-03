"""Prompt strategies and verbalizers for manager-sentiment extraction.

A :class:`PromptStrategy` bundles the three things that define a measure:

1. the ending appended to the filing text, per backend family
   (``causal`` reads the *next* token, ``mlm`` fills a ``[MASK]``);
2. the verbalizer -- which answer words count as positive and which as negative;
3. the sign convention, i.e. whether a higher polarity is good news.

The sign convention is stored explicitly because it is *not* obvious for the
risk measures: there, the "positive" word list contains *low-risk* words
("manageable", "negligible"), so a higher polarity means *less* perceived risk.
Carrying that as data keeps the downstream stages from flipping signs by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

MASK_PLACEHOLDER = "{mask}"

_TRAILING_PUNCT = re.compile(r"[\s,.;:!?)\]]+$")
_LEADING_PUNCT = re.compile(r"^[\s(\[]+")


def normalize_token(token: str) -> str:
    """Normalize a decoded model token for verbalizer lookup.

    Model tokens arrive with a leading space (``" optimistic"``) and sometimes
    trailing punctuation (``"modest:"``). Both are stripped so the verbalizers
    can be written as plain word lists.
    """
    token = _LEADING_PUNCT.sub("", token)
    token = _TRAILING_PUNCT.sub("", token)
    return token.lower()


def _verbalizer(positive: set[str], negative: set[str]) -> Mapping[str, frozenset[str]]:
    """Normalize and sanity-check a verbalizer definition."""
    pos = frozenset(normalize_token(w) for w in positive) - {""}
    neg = frozenset(normalize_token(w) for w in negative) - {""}
    overlap = pos & neg
    if overlap:
        raise ValueError(f"verbalizer words appear in both polarities: {sorted(overlap)}")
    return {"positive": pos, "negative": neg}


@dataclass(frozen=True)
class PromptStrategy:
    """A single sentiment measure."""

    name: str
    #: Backend family -> text appended after the filing. The ``mlm`` ending must
    #: contain :data:`MASK_PLACEHOLDER`.
    endings: Mapping[str, str]
    verbalizer: Mapping[str, frozenset[str]]
    #: ``True`` when a higher polarity is favourable news about the firm.
    higher_is_good: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if "mlm" in self.endings and MASK_PLACEHOLDER not in self.endings["mlm"]:
            raise ValueError(f"strategy {self.name!r}: mlm ending must contain {MASK_PLACEHOLDER}")
        for polarity in ("positive", "negative"):
            if polarity not in self.verbalizer:
                raise ValueError(f"strategy {self.name!r}: verbalizer lacks {polarity!r}")

    def ending(self, backend_kind: str, mask_token: str | None = None) -> str:
        """The prompt ending for one backend family, with the mask filled in."""
        if backend_kind not in self.endings:
            raise KeyError(
                f"strategy {self.name!r} has no ending for backend {backend_kind!r} "
                f"(available: {sorted(self.endings)})"
            )
        ending = self.endings[backend_kind]
        if MASK_PLACEHOLDER in ending:
            if not mask_token:
                raise ValueError(f"strategy {self.name!r}: backend {backend_kind!r} needs a mask token")
            ending = ending.replace(MASK_PLACEHOLDER, mask_token)
        return ending

    def build_prompt(self, text: str, backend_kind: str, mask_token: str | None = None) -> str:
        return f"{text}\n{self.ending(backend_kind, mask_token)}"


# ---------------------------------------------------------------------------
# Growth sentiment
# ---------------------------------------------------------------------------

_GROWTH_POSITIVE_CORE = {
    "optimistic", "confident", "positive", "encouraged", "excited",
    "enthusiastic", "hopeful", "pleased", "encouraging", "ambitious",
    "favorable", "assured", "strong", "good", "excellent", "outstanding",
    "healthy", "awesome", "great", "fantastic", "stable",
    "perfect", "solid", "profitable", "impressive", "reliable",
    "thriving", "sustainable", "exceptional", "promising", "bright", "attractive",
}
_GROWTH_POSITIVE_EXTRA = {
    "comfortable", "satisfied", "relaxed", "determined", "bullish",
    "upbeat", "convinced", "certain", "reassured",
}

_GROWTH_NEGATIVE_CORE = {
    "cautious", "concerned", "pessimistic", "negative",
    "uncomfortable", "uncertain", "unsure",
    "skeptical", "worried", "worrying", "discouraged", "anxious", "frustrated",
    "confused", "doubtful", "unsatisfied", "disappointed",
    "bad", "poor", "terrible", "risky", "weak", "dependent",
    "unstable", "unhealthy", "questionable", "suffering", "stressed",
    "unsustainable", "awful", "vulnerable",
    "mediocre", "horrible", "precarious", "declining", "worsening",
    "difficult", "limited", "challenged",
}
_GROWTH_NEGATIVE_EXTRA = {"dissatisfied", "nervous", "uneasy", "apprehensive"}

GROWTH = PromptStrategy(
    name="growth",
    endings={
        "causal": "Future growth prospects appear to be",
        "mlm": f"In one word, we are {MASK_PLACEHOLDER} regarding the future growth of our company.",
    },
    verbalizer=_verbalizer(
        positive=_GROWTH_POSITIVE_CORE | _GROWTH_POSITIVE_EXTRA,
        negative=_GROWTH_NEGATIVE_CORE | _GROWTH_NEGATIVE_EXTRA,
    ),
    higher_is_good=True,
    description="Management's outlook on future growth prospects.",
)

GROWTH_LEGACY_CAUSAL = PromptStrategy(
    name="growth_legacy_causal",
    endings={"causal": "Future growth prospects appear to be"},
    verbalizer=_verbalizer(positive=_GROWTH_POSITIVE_CORE, negative=_GROWTH_NEGATIVE_CORE),
    higher_is_good=True,
    description="Growth with the narrower word list used by the earlier Llama runs.",
)


# ---------------------------------------------------------------------------
# Financial risk
# ---------------------------------------------------------------------------

# NOTE on the sign: "positive" here means *low* risk, so polarity is still
# "good news for the firm" and higher_is_good stays True. Do not flip it
# downstream.

_RISK_LOW = {
    "balanced", "controlled", "declined", "declining", "decreased", "decreasing",
    "governed", "insignificant", "limited", "low", "lower", "manageable",
    "managed", "minimal", "minor", "mitig", "mitigated", "moder", "moderate",
    "moderated", "moderately", "modest", "negligible", "reduced", "reducing",
    "small", "slight",
}
_RISK_HIGH = {
    "considerable", "considerably", "extensive", "exacerbated", "extraordinarily",
    "extraordinary", "extreme", "elevated", "great", "greater", "growing",
    "heightened", "high", "higher", "highest", "huge", "immense", "increased",
    "increasing", "large", "rising", "serious", "severe", "significant",
    "significantly", "unprecedented", "substantial", "tremendous", "unacceptable",
    "sufficient"
}

FINANCIAL_RISK = PromptStrategy(
    name="financial_risk",
    endings={
        "causal": "In summary, I would say that the level of financial risk facing the company is",
        "mlm": f"In summary, the level of financial risk facing our company is {MASK_PLACEHOLDER}.",
    },
    verbalizer=_verbalizer(positive=_RISK_LOW, negative=_RISK_HIGH),
    higher_is_good=True,
    description="Perceived level of financial risk; higher polarity = lower risk.",
)


STRATEGIES: dict[str, PromptStrategy] = {
    strategy.name: strategy
    for strategy in (GROWTH, GROWTH_LEGACY_CAUSAL, FINANCIAL_RISK)
}


def get_strategy(name: str) -> PromptStrategy:
    try:
        return STRATEGIES[name]
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(STRATEGIES)}") from None
