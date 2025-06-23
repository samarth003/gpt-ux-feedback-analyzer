import pandas as pd
import re
from sentence_transformers import SentenceTransformer

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)
    df['cleaned'] = df['feedback'].apply(clean_text)
    return df

def vectorize_sentences(text_series):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_series.tolist(), show_progress_bar=True)
    return embeddings, model
