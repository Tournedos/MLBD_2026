import pandas as pd
import numpy as np
import re

# HELPER FUNCTIONS 
def split_sentences(text):
    if pd.isna(text):
        return []
    text = str(text).strip()
    if not text:
        return []
    sents = re.split(r"[.!?]+", text)
    return [s.strip() for s in sents if s.strip()]

def split_words(text):
    if pd.isna(text):
        return []
    text = str(text).lower()
    return re.findall(r"\b\w+\b", text)

def word_count(text):
    return len(split_words(text))

def char_count(text):
    if pd.isna(text):
        return 0
    return len(str(text))

def sentence_count(text):
    return len(split_sentences(text))

def avg_sentence_length(text):
    sents = split_sentences(text)
    if len(sents) == 0:
        return 0.0
    lengths = [len(split_words(s)) for s in sents]
    return float(np.mean(lengths)) if len(lengths) > 0 else 0.0

def avg_word_length(text):
    words = split_words(text)
    if len(words) == 0:
        return 0.0
    return float(np.mean([len(w) for w in words]))

def lexical_diversity(text):
    words = split_words(text)
    if len(words) == 0:
        return 0.0
    return len(set(words)) / len(words)

def comma_count(text):
    if pd.isna(text):
        return 0
    return str(text).count(",")

def semicolon_count(text):
    if pd.isna(text):
        return 0
    return str(text).count(";")

def colon_count(text):
    if pd.isna(text):
        return 0
    return str(text).count(":")

def exclamation_count(text):
    if pd.isna(text):
        return 0
    return str(text).count("!")

def question_count(text):
    if pd.isna(text):
        return 0
    return str(text).count("?")

def long_word_ratio(text, min_len=7):
    words = split_words(text)
    if len(words) == 0:
        return 0.0
    return sum(len(w) >= min_len for w in words) / len(words)