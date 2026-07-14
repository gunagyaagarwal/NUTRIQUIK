PREDICTION_KEYWORDS = [
    "predict", "risk of", "assess my", "chance of", "diagnose", "check my",
    "am i at risk", "do i have", "test for", "screen for", "evaluate my",
    "am i anemic", "check if i have", "what vitamin am i", "which deficiency",
]

ADVISORY_KEYWORDS = [
    "suggest", "recommend a", "recommend me", "recommend some", "can you recommend",
    "please recommend", "plan for", "diet for", "best foods", "how to improve",
    "what should i eat", "give me a", "foods for", "meal plan", "supplement for",
    "tips for", "how to boost", "how to reduce", "how to increase", "how to lose",
    "how to gain", "what to eat", "foods to", "help me", "how do i improve",
    "how to lower", "how to raise", "rich in", "high in", "low in", "diet plan",
]

DISEASE_CONTEXT_RULES = [
    (["diabetes", "diabetic", "diabities", "blood sugar", "glucose"], "diabetes"),
    (["anemia", "anemic", "hemoglobin", "iron deficiency"], "anemia"),
    (["heart", "cardiac", "cholesterol"], "heart"),
    (["kidney", "renal"], "kidney"),
    (["vitamin deficiency", "deficiency"], "vitamin_deficiency"),
    (["supplement", "creatine", "protein powder"], "supplement"),
    (["weight", "bmi", "obesity"], "weight"),
    (["diet recommendation"], "diet_recommendation"),
]

# Verbs that only imply a prediction when paired with a disease/condition word
# in the same query (e.g. "suggest my diabetes chances" or "potential of kidney
# disease"), since alone they're too generic ("suggest a meal plan" is advisory).
PREDICTION_COMBO_VERBS = ["predict", "suggest", "chances", "potential"]

DISEASE_TRIGGER_WORDS = [
    "diabetes", "diabetic", "diabities", "anemia",
    "vitamin deficiency", "kidney", "heart attack", "heart disease", "supplement",
    "diet recommendation",
]


def _has_prediction_combo(query_lower):
    has_verb = any(verb in query_lower for verb in PREDICTION_COMBO_VERBS)
    has_disease = any(word in query_lower for word in DISEASE_TRIGGER_WORDS)
    return has_verb and has_disease


def classify_intent(query):
    query_lower = query.lower()

    if any(keyword in query_lower for keyword in PREDICTION_KEYWORDS):
        return "prediction"

    if _has_prediction_combo(query_lower):
        return "prediction"

    if any(keyword in query_lower for keyword in ADVISORY_KEYWORDS):
        return "advisory"

    return "factual"


def detect_disease_context(query):
    query_lower = query.lower()

    for keywords, label in DISEASE_CONTEXT_RULES:
        if any(keyword in query_lower for keyword in keywords):
            return label

    return None


if __name__ == "__main__":
    test_queries = [
        "predict my diabetes risk",
        "am i at risk of anemia",
        "check my heart disease risk",
        "diet for kidney disease",
        "how to boost immunity",
        "supplement for muscle gain",
        "what is vitamin c",
        "what causes vitamin deficiency",
        "tips for weight loss",
    ]

    for query in test_queries:
        intent = classify_intent(query)
        disease = detect_disease_context(query)
        print(f"Query: '{query}'")
        print(f"  intent={intent}, disease_context={disease}")
