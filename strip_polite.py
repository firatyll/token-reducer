import re
import joblib

MODEL_PATH = "politeness_model.joblib" 
bundle = joblib.load(MODEL_PATH)
regex  = re.compile(bundle["regex_pat"], flags=re.I)

def strip_polite(sentence: str) -> str:
    out = regex.sub("", sentence)      
    out = re.sub(r"\s{2,}", " ", out)       
    return out.strip(" ,.?!")