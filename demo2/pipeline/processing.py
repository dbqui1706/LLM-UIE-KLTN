import re
from cleantext import clean
from ftfy import fix_text
from unidecode import unidecode
from typing import Optional


def clean_docling_output(text: str, aggressive: bool = False) -> str:
    if not text:
        return text

    # 1. Fix encoding issues với ftfy
    text = fix_text(text)

    # 2. Sử dụng cleantext library cho basic cleaning
    text = clean(
        text,
        fix_unicode=True,  # fix various unicode errors
        to_ascii=True,  # transliterate to closest ASCII representation
        lower=False,  # lowercase text
        no_line_breaks=False,  # fully strip line breaks as opposed to only normalizing them
        no_urls=True,  # replace all URLs with a special token
        no_emails=True,  # replace all email addresses with a special token
        no_phone_numbers=False,  # replace all phone numbers with a special token
        no_numbers=False,  # replace all numbers with a special token
        no_digits=False,  # replace all digits with a special token
        no_currency_symbols=False,  # replace all currency symbols with a special token
        no_punct=False,  # remove punctuations
        replace_with_punct="",  # instead of removing punctuations you may replace them
        replace_with_url="{{URL}}",
        replace_with_email="{{EMAIL}}>",
        lang="en"
    )
    if aggressive:
        text = unidecode(text)

    return text.strip()


def quick_clean_metadata(metadata: dict) -> dict:
    """Quick clean metadata"""
    if not metadata:
        return {}

    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, str) and value.strip():
            cleaned[key] = value.strip()
        elif isinstance(value, (int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, list) and value:
            cleaned[key] = [v for v in value if v]

    return cleaned
