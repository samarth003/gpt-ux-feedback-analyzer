from transformers import pipeline

# Load model once (can place outside the function for reuse)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def get_hf_summary(feedback_list):
    # Combine all lines into one string (models expect > min length)
    full_text = " ".join(feedback_list)

    # Hugging Face summarizers like longer inputs
    result = summarizer(full_text, max_length=130, min_length=30, do_sample=False)
    return result[0]['summary_text']
