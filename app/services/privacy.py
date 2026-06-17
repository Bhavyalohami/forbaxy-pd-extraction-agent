import re

PI_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(patient\s*name|name)\s*[:\-]", re.IGNORECASE),
    re.compile(r"\b(uhid|abha|opd|registration|reg\.?\s*no|mobile|phone|address)\b", re.IGNORECASE),
    re.compile(r"\b(age|gender|sex|patient\s*category)\s*[:\-]", re.IGNORECASE),
    re.compile(r"\b\d{10}\b"),
)


def contains_patient_information(text: str) -> bool:
    return any(pattern.search(text) for pattern in PI_PATTERNS)


def redact_patient_information(text: str) -> str:
    sanitized = text
    for pattern in PI_PATTERNS:
        sanitized = pattern.sub("[PI_REDACTED]", sanitized)
    return sanitized

