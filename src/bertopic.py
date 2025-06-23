from bertopic import BERTopic
from hdbscan import HDBSCAN

def cluster_with_bertopic(texts):
    # Define HDBSCAN model with smaller min_cluster_size
    hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=1)

    # Create BERTopic with the custom model
    topic_model = BERTopic(hdbscan_model=hdbscan_model, language="english", verbose=True)
    topics, probs = topic_model.fit_transform(texts)
    return topics, probs, topic_model

