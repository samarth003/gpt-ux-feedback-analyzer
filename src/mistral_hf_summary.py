from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=os.getenv("HF_API_TOKEN")  # Load from .env or system variable
)

def get_zephyr_summary(feedback_list):
    prompt = (
        "You are an expert UX product analyst. Based on the user feedback below, do the following:\n"
        "1. Summarize the main problems users are facing.\n"
        "2. List any feature requests mentioned or implied.\n"
        "3. Classify the tone as: Positive, Neutral, or Negative.\n\n"
        "Feedback:\n" + "\n".join(f"- {line}" for line in feedback_list)
    )

    try:
        response = client.text_generation(prompt, max_new_tokens=300, temperature=0.4)
        return response.strip()
    except Exception as e:
        return f"❌ Zephyr summarizer error: {str(e)}"


