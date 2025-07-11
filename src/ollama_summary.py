import subprocess

def get_ollama_summary(feedback_list):
    prompt = (
        "You are an expert UX product analyst. Based on the following user feedback:\n\n" +
        "\n".join(f"- {line}" for line in feedback_list) +
        "\n\n1. Summarize the main complaints.\n2. List feature requests.\n3. Classify tone: Positive, Neutral, or Negative."
    )

    try:
        result = subprocess.run(
            [r"C:\Users\samar\AppData\Local\Programs\Ollama\ollama.exe", "run", "mistral"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=60
        )
        return result.stdout.decode("utf-8").strip()

    except FileNotFoundError as fnf:
        return f"❌ Ollama not found — make sure it's in PATH: {str(fnf)}"

    except subprocess.TimeoutExpired:
        return "⚠️ Ollama summarizer timed out. Try summarizing fewer lines."

    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"
