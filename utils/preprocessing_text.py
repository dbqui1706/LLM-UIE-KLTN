import re
import string

def preprocessing_text(text):
    text = re.sub(r'\s+([,.!?:;])', r'\1', text)

    text = re.sub(r"\s+", " ", text)

    return text.strip().strip(string.punctuation).strip()
