from bertopic import BERTopic
from hdbscan import HDBSCAN

def get_hdbscan_model(text_count):
    """Return HDBSCAN model with tuned parameters based on dataset size."""
    if text_count < 15:
        return HDBSCAN(min_cluster_size=2, min_samples=1)
    elif text_count < 50:
        return HDBSCAN(min_cluster_size=5, min_samples=2)
    else:
        return HDBSCAN(min_cluster_size=10, min_samples=5)

def cluster_with_bertopic(texts):
    """
    Inputs: 
        texts (list of str): cleaned user feedback
    Returns: 
        topics: list of topic labels per input
        probs: confidence scores
        topic_model: full BERTopic model (for topic labels, keywords, etc.)
    """
    hdbscan_model = get_hdbscan_model(len(texts))
    topic_model = BERTopic(hdbscan_model=hdbscan_model, language="english", verbose=True)
    topics, probs = topic_model.fit_transform(texts)
    return topics, probs, topic_model


