import re

PREDICTION_KEYWORDS = [
    "predict", "risk of", "assess my", "chance of", "diagnose", "check my",
    "am i at risk", "do i have", "test for", "screen for", "evaluate my",
    "am i anemic", "am i diabetic", "check if i have", "what vitamin am i", "which deficiency",
]

# "am i diabetic/hypertensive/anemic/..." and "is my blood sugar/BP/cholesterol normal"
# style self-diagnosis questions, generalized beyond the literal phrases above.
_AM_I_CONDITION_PATTERN = re.compile(r"\bam i \w+\b")
_IS_MY_X_NORMAL_PATTERN = re.compile(r"\bis my [\w\s]+ (normal|high|low)\b")

ADVISORY_KEYWORDS = [
    "suggest", "recommend a", "recommend me", "recommend some", "can you recommend",
    "please recommend", "plan for", "diet for", "best foods", "how to improve",
    "what should i eat", "give me a", "foods for", "meal plan", "supplement for",
    "tips for", "how to boost", "how to reduce", "how to increase", "how to lose",
    "how to gain", "what to eat", "foods to", "help me", "how do i improve",
    "how to lower", "how to raise", "rich in", "high in", "low in", "diet plan",
    # symptom/prevention/severity/supplement-advice phrasing — these want practical
    # guidance (advisory top-5 + trust scoring), not a single strict factual lookup.
    "symptoms of", "supplements help", "supplement help", "prevent", "severity",
    # A request mentioning "recipe(s)" always wants food preparation guidance, never
    # a definitional lookup — critical since the factual track explicitly excludes
    # Recipes-category documents (a recipe is never the right answer to "what is X"),
    # so a recipe request classified as factual would get zero recipes back, always.
    # "recipi" (not just "recipe") also catches the common misspelling "recipies".
    "recipe", "recipi",
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

# Verbs that only imply a prediction when paired with a disease/condition word in the
# same query (e.g. "predict my diabetes chances" or "potential of kidney disease").
# "suggest" is deliberately excluded — it's overwhelmingly an advisory verb ("suggest
# a recipe", "suggest diabetic-friendly meals"), and including it here misrouted
# queries like "suggest diabetic, kidney friendly recipes" into a risk-assessment
# form instead of the advisory recipe track.
PREDICTION_COMBO_VERBS = ["predict", "chances", "potential"]

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

    if _AM_I_CONDITION_PATTERN.search(query_lower) or _IS_MY_X_NORMAL_PATTERN.search(query_lower):
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
