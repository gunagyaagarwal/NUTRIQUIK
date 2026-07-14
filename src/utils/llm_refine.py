import os

GEMINI_MODEL = "gemini-2.0-flash"


def _get_gemini_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(GEMINI_MODEL)
    except Exception:
        return None


def refine_answer(raw_text, query):
    model = _get_gemini_model()
    if model is None:
        return raw_text

    prompt = (
        f"You are a nutrition expert. User asked: '{query}'. "
        f"Based ONLY on this text, answer in 2-3 clear sentences. "
        f"End with health disclaimer. Reference: {raw_text}"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return raw_text


def summarize_results(results, query):
    model = _get_gemini_model()
    if model is None:
        return "Summary unavailable"

    top_texts = [r.get("content", "") for r in results[:3]]
    prompt = (
        f"You are a nutrition expert. User asked: '{query}'. "
        f"Summarize the following results into one clear paragraph. "
        f"End with health disclaimer.\n\n" + "\n\n".join(top_texts)
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "Summary unavailable"


if __name__ == "__main__":
    print(refine_answer("Vitamin C is an antioxidant that supports immune function.", "what is vitamin c"))
