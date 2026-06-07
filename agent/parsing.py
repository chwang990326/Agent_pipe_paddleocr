"""Parsing helpers for pipe marks recognized by OCR."""

from __future__ import annotations

import re

MATERIAL_PATTERN = re.compile(r"\b(304L|316L|Q\d{3}[A-Z]?|[A-Z]{1,4}\d{1,4}[A-Z]?)\b", re.I)
DN_PATTERN = re.compile(r"\bDN\s*-?\s*(\d{2,5})\b", re.I)
LENGTH_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*M\b", re.I)


def normalize_material_id(text: str | None) -> str:
    if not text:
        return ""
    normalized = text.strip().upper()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.replace("_", "-")
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized


def parse_pipe_text(text: str | None) -> dict[str, str | None]:
    normalized = normalize_material_id(text)
    material_match = MATERIAL_PATTERN.search(normalized)
    dn_match = DN_PATTERN.search(normalized)
    length_match = LENGTH_PATTERN.search(normalized)

    diameter = f"DN{dn_match.group(1)}" if dn_match else None
    length = f"{length_match.group(1)}m" if length_match else None

    return {
        "material_id": normalized or None,
        "material": material_match.group(1).upper() if material_match else None,
        "diameter": diameter,
        "length": length,
    }
