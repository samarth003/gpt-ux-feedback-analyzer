import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.preprocessing import load_and_clean_data, vectorize_sentences
from src.clustering import cluster_feedback

st.title("GPT-Powered UX Feedback Analyzer")

uploaded_file = st.file_uploader("Upload a CSV file with feedback", type="csv")

if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    st.write("Cleaned Feedback Sample:")
    st.dataframe(df[['feedback', 'cleaned']].head())

    embeddings, model = vectorize_sentences(df['cleaned'])
    labels, _ = cluster_feedback(embeddings)

    df['cluster'] = labels

    num_clusters = df['cluster'].nunique()
    for i in range(num_clusters):
        st.subheader(f"Cluster {i}")
        cluster_texts = df[df['cluster'] == i]['feedback'].tolist()[:5]
        for text in cluster_texts:
            st.markdown(f"- {text}")