import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.preprocessing import load_and_clean_data, vectorize_sentences
from src.clustering import cluster_feedback
from src.bertopic import cluster_with_bertopic
from src.gpt_summary import group_feedback_by_topic, build_prompt_for_topic, get_gpt_insight
from src.hf_summary import get_hf_summary
from src.mistral_hf_summary import get_zephyr_summary
from src.ollama_summary import get_ollama_summary

st.title("GPT-Powered UX Feedback Analyzer")

uploaded_file = st.file_uploader("Upload a CSV file with feedback", type="csv")
use_bertopic = st.sidebar.checkbox("Use BERTopic instead of KMeans", value=True)

def display_structured_summary(summary):
    if "Main Complaints" in summary:
        try:
            parts = summary.split("Feature Requests:")
            complaints = parts[0].replace("Main Complaints:", "").strip()
            rest = parts[1].split("Tone:")
            features = rest[0].strip()
            tone = rest[1].strip()

            st.markdown("### 🔴 Main Complaints")
            for line in complaints.split("\n"):
                if line.strip():
                    st.markdown(f"- {line.strip()}")

            st.markdown("### 🟢 Feature Requests")
            for line in features.split("\n"):
                if line.strip():
                    st.markdown(f"- {line.strip()}")

            st.markdown(f"### 🟡 Tone: **{tone}**")
        except Exception:
            st.markdown(summary)
    else:
        st.markdown(summary)

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

    topic_groups = group_feedback_by_topic(df, text_col='feedback', topic_col='topic')
    model_choice = st.selectbox("Choose summarizer", ["HuggingFace-BART", "OpenAI-GPT", "Ollama-MISTRAL"])

    for topic_id, feedbacks in topic_groups.items():
        if topic_id == -1:
            continue
        st.subheader(f"GPT Summary for Topic {topic_id}")
        if len(feedbacks) < 2:
            st.info("Not enough data to summarize this topic.")
            continue
        summary_key = f"summary_{topic_id}"
        if st.button(f"Summarize Topic {topic_id}"):
            if summary_key not in st.session_state:
                with st.spinner(f"Summarizing Topic {topic_id}..."):
                    if model_choice == "HuggingFace-BART":
                        st.session_state[summary_key] = get_hf_summary(feedbacks)
                    elif model_choice == "OpenAI-GPT":
                        prompt = build_prompt_for_topic(feedbacks)
                        st.session_state[summary_key] = "[OpenAI summary disabled \u2013 uncomment call to enable]"
                        # st.session_state[summary_key] = get_gpt_insight(prompt)
                    elif model_choice == "Ollama-MISTRAL":
                        st.session_state[summary_key] = get_ollama_summary(feedbacks)

            with st.expander(f"Topic {topic_id} Summary"):
                display_structured_summary(st.session_state[summary_key])