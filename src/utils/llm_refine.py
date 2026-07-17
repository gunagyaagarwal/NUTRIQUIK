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


def _fallback_summary(results):
    """Non-AI summary built directly from retrieved content, used whenever Gemini
    isn't configured/available — so the summarize button always produces something
    useful instead of a dead-end "Summary unavailable" message."""
    parts = []
    for r in results[:3]:
        title = r.get("title", "").strip()
        first_sentence = r.get("content", "").strip().split(". ")[0].strip()
        if first_sentence and not first_sentence.endswith("."):
            first_sentence += "."
        parts.append(f"**{title}** — {first_sentence}" if title and first_sentence else title)
    if not parts:
        return "No results available to summarize."
    return (
        "\n\n".join(p for p in parts if p)
        + "\n\n*(Quick summary from the retrieved results — set a GEMINI_API_KEY for an AI-written summary instead.)*"
    )


def summarize_results(results, query):
    model = _get_gemini_model()
    if model is None:
        return _fallback_summary(results)

    top_texts = [r.get("content", "") for r in results[:3]]
    prompt = (
        f"You are a nutrition expert. User asked: '{query}'. "
        f"Summarize the following results into one clear paragraph. "
        f"End with health disclaimer.\n\n" + "\n\n".join(top_texts)
    )
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        return text if text else _fallback_summary(results)
    except Exception:
        return _fallback_summary(results)


if __name__ == "__main__":
    print(refine_answer("Vitamin C is an antioxidant that supports immune function.", "what is vitamin c"))
