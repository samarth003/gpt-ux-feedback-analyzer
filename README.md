# GPT-Powered UX Feedback Analyzer

An AI-powered tool that helps product and UX teams analyze large volumes of user feedback using clustering and GPT-based summarization.

## 🧠 What It Does
- Upload a CSV of user reviews or support tickets
- Automatically groups similar feedback using KMeans or BERTopic
- GPT summarizes key pain points and feature requests
- PM-ready insights in seconds

## 📁 Project Structure

- `data/` — Sample feedback CSV
- `src/` — Core logic (preprocessing, clustering, GPT prompts)
- `app/` — Streamlit frontend
- `notebooks/` — For experimentation
- `requirements.txt` — Dependencies

## 🚀 How to Run

1. Clone the repo
    git clone https://github.com/your-username/gpt-ux-feedback-analyzer.git
    cd gpt-ux-feedback-analyzer

2. Install dependencies
    pip install requirements.txt

3. Launch the app
    streamlit run app/streamlit_app.py