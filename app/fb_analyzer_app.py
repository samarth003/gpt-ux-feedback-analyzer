import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.preprocessing import load_and_clean_data, vectorize_sentences
from src.clustering import cluster_feedback
from src.bertopic import cluster_with_bertopic

st.title("GPT-Powered UX Feedback Analyzer")

uploaded_file = st.file_uploader("Upload a CSV file with feedback", type="csv")
use_bertopic = st.sidebar.checkbox("Use BERTopic instead of KMeans", value=True)

if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    st.write("Cleaned Feedback Sample:")
    st.dataframe(df[['feedback', 'cleaned']].head())

    if use_bertopic:
        topics, probs, topic_model = cluster_with_bertopic(df['cleaned'].tolist())
        df['topic'] = topics
        topic_info = topic_model.get_topic_info()
        num_topics = len(topic_info)

        for topic_num in topic_info['Topic'].tolist():
            if topic_num == -1:
                continue  # -1 is usually "outlier" cluster
            st.subheader(f"Topic {topic_num}: {topic_model.get_topic(topic_num)}")
            topic_feedbacks = df[df['topic'] == topic_num]['feedback'].tolist()[:5]
            for fb in topic_feedbacks:
                st.markdown(f"- {fb}")
    else:    
        embeddings, model = vectorize_sentences(df['cleaned'])
        labels, _ = cluster_feedback(embeddings)
        df['cluster'] = labels
        num_clusters = df['cluster'].nunique()
        for i in range(num_clusters):
            st.subheader(f"Cluster {i}")
            cluster_texts = df[df['cluster'] == i]['feedback'].tolist()[:5]
            for text in cluster_texts:
                st.markdown(f"- {text}")