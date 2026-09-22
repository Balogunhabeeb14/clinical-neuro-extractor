"""Shared cognitive-domain header lexicon.

Neuropsych reports and assessment score sheets both group content under
domain headers (e.g. "MEMORY FUNCTIONS" in a report, "Memory" in a score
sheet). This lexicon lets both parsers recognize the same domains under
their different real-world spellings.

The list is not exhaustive - extend ``DOMAIN_ALIASES`` as new header
spellings turn up in real documents.
"""

from __future__ import annotations

# canonical domain slug -> raw header spellings seen in documents (any case)
DOMAIN_ALIASES: dict[str, list[str]] = {
    "intellectual_functioning": [
        "intellectual functions",
        "intellectual functioning",
        "general intellectual functioning",
        "general intellectual function",
    ],
    "memory": [
        "memory functions",
        "memory function",
        "memory",
    ],
    "executive_functioning": [
        "executive functions",
        "executive function",
        "executive functioning",
    ],
    "language": [
        "language",
        "language functions",
    ],
    "attention": [
        "attention",
        "attention and concentration",
        "attention functions",
    ],
    "visuospatial": [
        "visuospatial",
        "visuospatial functions",
        "visuoperceptual",
        "visuoperceptual functions",
    ],
    "processing_speed": [
        "processing speed",
    ],
    "mood": [
        "mood",
        "emotional functioning",
        "mood and emotional functioning",
    ],
    "personality": [
        "personality",
    ],
    "behaviour": [
        "behaviour",
        "behavior",
        "behavioural functioning",
        "behavioral functioning",
    ],
}

# raw header (lowercased, whitespace-normalized) -> canonical domain slug
_ALIAS_LOOKUP: dict[str, str] = {
    alias: slug for slug, aliases in DOMAIN_ALIASES.items() for alias in aliases
}


def canonical_domain(line: str) -> str | None:
    """Return the canonical domain slug for a standalone header line, if any."""
    normalized = " ".join(line.strip().lower().rstrip(":").split())
    return _ALIAS_LOOKUP.get(normalized)
