import subprocess

def get_ollama_summary(feedback_list):
    prompt = (
        "You are an expert UX product analyst. Based on the following user feedback:\n\n"
        + "\n".join(f"- {line}" for line in feedback_list)
        + "\n\nTasks:\n"
        "1. Summarize the main complaints.\n"
        "2. List feature requests.\n"
        "3. Classify the tone: Positive, Neutral, or Negative."
    )

    try:
        # First attempt with a normal timeout
        result = run_subprocess(structured_prompt=prompt, timeout=60)
        return result.stdout.decode("utf-8").strip()

    except subprocess.TimeoutExpired:
        # Retry once with a longer timeout
        try:
            result = run_subprocess(structured_prompt=prompt, timeout=120)
            return "⏳ Initial call timed out, but retry succeeded:\n\n" + result.stdout.decode("utf-8").strip()
        except Exception as retry_e:
            return f"❌ Retry also failed: {str(retry_e)}"

    except FileNotFoundError as fnf:
        return f"❌ Ollama not found: {str(fnf)}"

    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

def run_subprocess(structured_prompt, timeout=60):
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=structured_prompt.encode("utf-8"),
        capture_output=True,
        timeout=timeout
    )
    return result    