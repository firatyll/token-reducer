import re

POLITE_PATTERNS = [
    r"\b(?:thanks|thank you|thx|much appreciated|cheers)\b",
    r"\b(?:have a (nice|great) (day|evening|weekend))\b",
    r"\b(?:best regards|kind regards|regards|sincerely)\b",
    r"\b(?:hello|hi|hey|good (morning|afternoon|evening))\b",
    r"\b(?:please|could you|would you (mind)?|if possible)\b",
]
def trim(text: str, patterns=POLITE_PATTERNS) -> str:

    cleaned = text
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned) 
    cleaned = re.sub(r"\s,|,\s,", ",", cleaned) 
    return cleaned.strip(" ,")
