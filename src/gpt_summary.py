import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set in your environment

def get_gpt_insight(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" if using free tier
        messages=[
            {"role": "system", "content": "You are an expert product analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )
    return response['choices'][0]['message']['content']


def group_feedback_by_topic(df, text_col='feedback', topic_col='topic'):
    """Returns a dict: {topic_num: [feedback1, feedback2, ...]}"""
    grouped = {}
    for topic in df[topic_col].unique():
        group = df[df[topic_col] == topic][text_col].tolist()
        grouped[topic] = group
    return grouped

def build_prompt_for_topic(feedback_list):
    joined_feedback = "\n".join(f"- {line}" for line in feedback_list)
    prompt = f"""
    You are a UX product analyst. Based on the user feedback below, perform three tasks:

    {joined_feedback}

    1. Summarize what the users are complaining about or requesting.
    2. List any features users want (explicit or implied).
    3. Classify the overall tone as: Positive, Neutral, or Negative.
    """
    return prompt
