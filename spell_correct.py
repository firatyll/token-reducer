import re
from spellchecker import SpellChecker

spell = SpellChecker(distance=3)

def normalize_repetitions(word: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", word)

def correct_spell(sentence: str) -> str:
    tokens = sentence.split()           
    corrected = []
    for tok in tokens:
        norm = normalize_repetitions(tok)
        if norm.lower() in spell:
            corrected.append(tok) 
        else:
            corr = spell.correction(norm)
            corrected.append(corr if corr else tok)

    return " ".join(corrected)
