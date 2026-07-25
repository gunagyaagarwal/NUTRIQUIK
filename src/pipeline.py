"""Pure-logic query pipeline: query guard -> intent router -> factual/advisory/
prediction/diet_form tracks. Ported verbatim from the former app.py (Streamlit UI)
so the frontend swap changes presentation only, not behavior, thresholds, or scoring.
No UI framework dependency here - only src/guard, src/ir, src/ml.
"""
import os
import re
import sys
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.guard.query_guard import run_query_guard  # noqa: E402
from src.guard.intent_router import classify_intent, detect_disease_context  # noqa: E402
from src.ir.bm25 import BM25Index, load_documents  # noqa: E402
from src.ir.vector_index import VectorIndex  # noqa: E402
from src.ir.hybrid import hybrid_search  # noqa: E402
from src.ml.predict import score_documents, rank_by_trust, apply_guardrail  # noqa: E402

# Below this raw semantic-similarity score, the "best" retrieval match is judged too
# weak to be a real answer (e.g. corpus has no doc for the topic and BM25/vector
# search just returned the least-irrelevant thing) — show a "not available" message
# instead of presenting a misleading result.
FACTUAL_RELEVANCE_THRESHOLD = 0.35

# Composite trust-score cutoff for the advisory guardrail. Recalibrated alongside the
# bm25_score normalization fix in src/ml/predict.py: the old 0.5 cutoff was tuned
# against a scoring formula that batch-relative-normalized already-bounded [0, 1]
# features, artificially inflating scores. Under the corrected (more honest) scale,
# an adversarial/off-domain query now tops out around 0.23, while genuinely relevant
# advisory results typically run 0.3-0.6+ — 0.30 keeps clear separation from junk
# while no longer rejecting most real, on-topic content.
ADVISORY_TRUST_THRESHOLD = 0.30

# Ordered longest-first so "vitamin b12"/"vitamin b6" are matched specifically
# rather than only ever matching the generic "vitamin b" prefix.
NUTRIENT_QUERY_TERMS = [
    "vitamin b12", "vitamin b6", "vitamin a", "vitamin b", "vitamin c", "vitamin d",
    "vitamin e", "vitamin k", "calcium", "iron", "magnesium", "zinc", "potassium",
    "protein", "fiber", "fibre",
]

# Requires actual planning/recommendation intent, not just a nutrient word appearing
# anywhere in the query — "foods rich in protein" or "recipes containing low sodium"
# are plain advisory/factual lookups, not a request to build a personalized diet plan.
PERSONALIZED_DIET_PHRASES = [
    "diet plan", "meal plan", "what should i eat", "suggest meals", "suggest a meal plan",
    "suggest my meals", "personalized diet", "custom diet", "custom meal plan",
    "plan my diet", "plan my meals", "design my diet", "create a diet plan",
    "create my diet", "recommend a diet plan", "recommend my diet", "build me a diet",
]
# "more protein"/"less sodium" only counts as a diet-planning signal when it's also
# talking about diet/meals/food, not just any comparison question.
_MORE_LESS_PATTERN = re.compile(r"\b(more|less)\s+\w+\b.*\b(diet|meal|food|foods)\b")


def is_personalized_diet_request(query):
    query_lower = query.lower()
    if any(phrase in query_lower for phrase in PERSONALIZED_DIET_PHRASES):
        return True
    return bool(_MORE_LESS_PATTERN.search(query_lower))


# BM25/category-relevance/term-overlap all rely on exact token matches, so a
# misspelling like "recipies" scores near-zero on several trust features at once
# even against a genuinely relevant document — normalizing common misspellings
# before retrieval fixes the root cause instead of patching each symptom.
_QUERY_SPELLING_FIXES = [
    (re.compile(r"\brecipies\b", re.IGNORECASE), "recipes"),
    (re.compile(r"\brecipie\b", re.IGNORECASE), "recipe"),
    (re.compile(r"\brecepies\b", re.IGNORECASE), "recipes"),
    (re.compile(r"\brecepie\b", re.IGNORECASE), "recipe"),
    (re.compile(r"\brecipy\b", re.IGNORECASE), "recipe"),
    (re.compile(r"\bchickengunia\b", re.IGNORECASE), "chikungunya"),
    (re.compile(r"\bchickungunya\b", re.IGNORECASE), "chikungunya"),
    (re.compile(r"\bchickungunia\b", re.IGNORECASE), "chikungunya"),
]


def normalize_query_spelling(query):
    normalized = query
    for pattern, replacement in _QUERY_SPELLING_FIXES:
        normalized = pattern.sub(replacement, normalized)
    return normalized


# Every Vitamins & Minerals / Deficiency Diseases doc is written with this
# consistent "Deficiency symptoms: ..." sentence — when the user specifically
# asks about symptoms, extract just that sentence instead of showing the whole
# document (benefits, food sources, RDA, etc. they didn't ask for).
_SYMPTOM_SENTENCE_PATTERN = re.compile(r"Deficiency symptoms:[^.]*\.", re.IGNORECASE)


def extract_symptom_focused_answer(query, content):
    if "symptom" not in query.lower():
        return content
    match = _SYMPTOM_SENTENCE_PATTERN.search(content)
    return match.group(0).strip() if match else None


def find_named_vitamin_mineral_doc(query, index):
    """Direct name match for a Vitamins & Minerals doc mentioned in the query.
    Needed because semantic ranking alone is unreliable here: for an ambiguous
    query like "vitamin a deficiency" or "vitamin b deficiency", the generic
    Vitamin Deficiency Overview (or even an unrelated doc like Anemia) can score
    higher on vector similarity than the specific vitamin's own page, purely
    from sharing more "deficiency"-flavored vocabulary — not because it's a
    better answer to "the user asked about vitamin A specifically"."""
    query_lower = query.lower()
    candidates = [
        (doc_id, meta.get("title", "")) for doc_id, meta in index.doc_metadata.items()
        if meta.get("category") == "Vitamins & Minerals"
    ]
    candidates.sort(key=lambda x: len(x[1]), reverse=True)
    for doc_id, title in candidates:
        if title and re.search(rf"\b{re.escape(title.lower())}\b", query_lower):
            return doc_id
    return None


# Deficiency symptom phrases per Vitamins & Minerals doc, taken directly from each
# doc's own "Deficiency symptoms: ..." sentence (plus a few natural-phrasing variants,
# e.g. "gums bleed" for "bleeding gums"). Used to identify the SPECIFIC nutrient a
# user's described symptoms point to, rather than falling back to the generic Vitamin
# Deficiency Overview — the same problem find_named_vitamin_mineral_doc solves for
# queries that name a vitamin directly, but for queries that only describe symptoms
# ("I have bleeding gums and fatigue, what am I deficient in?").
VITAMIN_DEFICIENCY_SYMPTOMS = {
    "vm_vitamin_a": ["night blindness", "dry eyes", "dry skin", "frequent infections"],
    "vm_vitamin_b1": ["fatigue", "irritability", "nerve damage", "muscle weakness"],
    "vm_vitamin_b2": ["cracked lips", "sore throat", "skin disorders", "eye fatigue"],
    "vm_vitamin_b3": ["fatigue", "skin rashes", "digestive issues", "mental confusion"],
    "vm_vitamin_b5": ["fatigue", "irritability", "numbness", "tingling"],
    "vm_vitamin_b12": [
        "fatigue", "weakness", "numbness", "tingling", "memory problems",
        "forgetful", "pale skin",
    ],
    "vm_vitamin_b6": ["anemia", "skin rashes", "confusion", "weakened immunity"],
    "vm_vitamin_b7": ["hair thinning", "thinning hair", "brittle nails", "skin rashes"],
    "vm_vitamin_b9": ["fatigue", "weakness", "poor concentration", "birth defects"],
    "vm_vitamin_c": [
        "bleeding gums", "gums bleed", "gum bleeding", "bleeding gum",
        "slow wound healing", "wounds heal slowly", "joint pain",
        "easy bruising", "bruise easily",
    ],
    "vm_vitamin_d": ["bone pain", "bones ache", "muscle weakness", "fracture"],
    "vm_vitamin_e": ["muscle weakness", "vision problems", "immune issues"],
    "vm_vitamin_k": ["easy bruising", "bruise easily", "excessive bleeding", "poor bone health"],
    "vm_calcium": ["muscle cramps", "brittle nails", "numbness", "fracture"],
    "vm_iron": [
        "fatigue", "pale skin", "weakness", "shortness of breath",
        "cold hands", "cold feet",
    ],
    "vm_magnesium": ["muscle cramps", "fatigue", "irregular heartbeat", "numbness"],
    "vm_zinc": ["hair loss", "losing hair", "delayed wound healing", "weakened immunity", "loss of appetite"],
    "vm_potassium": ["muscle weakness", "cramps", "irregular heartbeat", "fatigue"],
    "vm_sodium": ["nausea", "muscle cramps", "confusion", "low blood pressure"],
    "vm_phosphorus": ["bone pain", "weakness", "loss of appetite"],
    "vm_iodine": ["goiter", "fatigue", "weight gain", "developmental delays"],
    "vm_selenium": ["muscle weakness", "fatigue", "weakened immunity", "thyroid problems"],
    "vm_copper": ["fatigue", "pale skin", "weakened immunity", "brittle bones"],
    "vm_manganese": ["bone malformation", "weakness", "impaired growth"],
    "vm_chromium": ["impaired glucose tolerance", "weight loss"],
    "vm_fluoride": ["tooth decay"],
}

# Direct nutrient -> corresponding Deficiency Diseases doc_id, for the few nutrients
# that actually have a dedicated named disease in the corpus. Deliberately not
# populated for nutrients without one (e.g. Vitamin B2, Zinc) — no link is better
# than a wrong one.
VITAMIN_TO_DEFICIENCY_DISEASE = {
    "vm_vitamin_a": "dis_008",   # Night Blindness / Xerophthalmia
    "vm_vitamin_b1": "dis_004",  # Beriberi
    "vm_vitamin_b3": "dis_005",  # Pellagra
    "vm_vitamin_b9": "dis_002",  # Anemia (folate deficiency)
    "vm_vitamin_b12": "dis_002", # Anemia (pernicious anemia)
    "vm_vitamin_c": "dis_001",   # Scurvy
    "vm_vitamin_d": "dis_003",   # Rickets
    "vm_calcium": "dis_003",     # Rickets
    "vm_iron": "dis_002",        # Anemia
    "vm_iodine": "dis_006",      # Goiter and Iodine Deficiency Disorders
}

_phrase_doc_counts = Counter(
    phrase for phrases in VITAMIN_DEFICIENCY_SYMPTOMS.values() for phrase in set(phrases)
)


def _build_symptom_pattern(phrase):
    """Match a symptom phrase as written ("hair loss") AND as one fused word
    ("hairloss") — real users type both. Only multi-word phrases get a fused
    variant; single words already match as-is."""
    spaced = re.escape(phrase)
    if " " not in phrase:
        return re.compile(rf"\b{spaced}\b", re.IGNORECASE)
    fused = re.escape(phrase.replace(" ", ""))
    return re.compile(rf"\b(?:{spaced}|{fused})\b", re.IGNORECASE)


_SYMPTOM_PATTERNS = {phrase: _build_symptom_pattern(phrase) for phrase in _phrase_doc_counts}

# Gate on actual deficiency-inquiry framing so this override doesn't hijack unrelated
# queries that happen to mention a generic symptom word like "fatigue" for other
# reasons (e.g. "foods that fight fatigue" — an advisory query, not a symptom lookup).
_DEFICIENCY_QUESTION_PATTERN = re.compile(r"deficien|which vitamin|what vitamin", re.IGNORECASE)

# A single common symptom shared across many nutrients (e.g. bare "fatigue", present
# in ~10 of the docs above) shouldn't be enough to confidently pin down one specific
# vitamin — this minimum total weight roughly requires either one fairly distinctive
# symptom or several overlapping ones before committing to a single nutrient.
_SYMPTOM_MATCH_MIN_WEIGHT = 0.3

# Generic "pain in my legs/arms/muscles/bones/joints" phrasing doesn't literally
# contain any single listed symptom phrase (e.g. "bone pain") even though it's one
# of the most common everyday ways people describe the classic Vitamin D deficiency
# presentation (diffuse bone/muscle pain) — matched separately via a body-part +
# pain-word regex (covering both "leg pain"/"legs hurt" AND the fused "legpain")
# rather than trying to enumerate every phrasing combination.
_PAIN_WORDS = r"(?:pain|ache|aches|aching|hurt|hurts|hurting|sore|sores)"
_BODY_PARTS = r"(?:leg|legs|arm|arms|limb|limbs|muscle|muscles|bone|bones|joint|joints|body)"
_GENERALIZED_PAIN_PATTERN = re.compile(
    rf"\b{_PAIN_WORDS}\b[^.]{{0,25}}\b{_BODY_PARTS}\b"
    rf"|\b{_BODY_PARTS}\b[^.]{{0,25}}\b{_PAIN_WORDS}\b"
    rf"|\b{_BODY_PARTS}{_PAIN_WORDS}\b",
    re.IGNORECASE,
)


def find_symptom_matched_vitamin_doc(query):
    if not _DEFICIENCY_QUESTION_PATTERN.search(query):
        return None
    # Hyphens collapse to spaces so "hair-loss" is handled by the same "hair loss"
    # match as the plain spaced phrasing, on top of the separately-checked fully
    # fused "hairloss" variant in _SYMPTOM_PATTERNS.
    normalized = query.replace("-", " ")
    scores = {}
    for doc_id, phrases in VITAMIN_DEFICIENCY_SYMPTOMS.items():
        matched = {p for p in phrases if _SYMPTOM_PATTERNS[p].search(normalized)}
        if matched:
            scores[doc_id] = sum(1.0 / _phrase_doc_counts[p] for p in matched)
    if _GENERALIZED_PAIN_PATTERN.search(normalized):
        scores["vm_vitamin_d"] = scores.get("vm_vitamin_d", 0.0) + 1.0
    if not scores:
        return None
    best_doc_id = max(scores, key=scores.get)
    return best_doc_id if scores[best_doc_id] >= _SYMPTOM_MATCH_MIN_WEIGHT else None


# A query naming two things ("egg or chicken", "milk vs paneer", "which is better ...")
# wants BOTH sides addressed — the factual track otherwise always truncates to a single
# top-ranked document, which silently drops the second thing being asked about and
# answers as if only one item was ever mentioned.
_COMPARISON_PATTERN = re.compile(
    r"\b(vs\.?|versus|compare[d]?(?:\s+to)?|which is (?:better|healthier)|better than|"
    r"worse than|difference between)\b",
    re.IGNORECASE,
)
_OR_COMPARISON_PATTERN = re.compile(
    r"\b(better|healthier|healthy|good for (?:you|health)|worse)\b.*\bor\b|"
    r"\bor\b.*\b(better|healthier)\b",
    re.IGNORECASE,
)


def is_comparison_query(query):
    query_lower = query.lower()
    return bool(_COMPARISON_PATTERN.search(query_lower) or _OR_COMPARISON_PATTERN.search(query_lower))


# Direct overrides for topics where "veg"/"non-veg" style phrasing reliably ranks the
# wrong Diet Plans doc via vector similarity (both docs share almost all the same
# "diet"/"nutrition" vocabulary, so the ranking signal is weak) — checked non-veg
# first since "non-vegetarian" also contains the word "vegetarian".
_NAMED_DIET_OVERRIDES = [
    (re.compile(r"\bnon[- ]?veg(?:etarian)?\b", re.IGNORECASE), "diet_002"),
    (re.compile(r"\bveg(?:etarian)?\b", re.IGNORECASE), "diet_001"),
]


def find_named_diet_doc(query):
    for pattern, doc_id in _NAMED_DIET_OVERRIDES:
        if pattern.search(query):
            return doc_id
    return None


_bm25_index = None
_bm25_documents = None
_vector_index = None


def get_bm25_index():
    global _bm25_index, _bm25_documents
    if _bm25_index is None:
        _bm25_documents = load_documents()
        _bm25_index = BM25Index()
        _bm25_index.build(_bm25_documents)
    return _bm25_index, _bm25_documents


def get_vector_index():
    global _vector_index
    if _vector_index is None:
        documents = load_documents()
        _vector_index = VectorIndex()
        _vector_index.load_or_build(documents)
    return _vector_index


def run_full_pipeline(query):
    try:
        guard_result = run_query_guard(query)
    except Exception:
        guard_result = {"allowed": True}

    if not guard_result.get("allowed", True):
        return {
            "status": "REJECTED_BY_GUARD",
            "reason": guard_result.get("message", "Blocked by guardrail."),
            "guard_reason": guard_result.get("reason"),
        }

    try:
        intent = classify_intent(query)
    except Exception:
        intent = "factual"

    if intent == "prediction":
        # "which deficiency"/"what vitamin am i" style phrasing classifies as
        # prediction intent, but a user describing actual symptoms in plain text
        # ("I have bleeding gums and fatigue...") wants a direct answer naming the
        # specific nutrient, not a lab-value risk-assessment form asking for numbers
        # they don't have. Only fall through to the form when no symptom match
        # is confident enough to answer directly.
        if find_symptom_matched_vitamin_doc(query) is None:
            try:
                disease = detect_disease_context(query)
            except Exception:
                disease = None
            return {"status": "SUCCESS", "intent": "prediction", "disease": disease}
        intent = "factual"

    if intent == "advisory" and is_personalized_diet_request(query):
        return {"status": "SUCCESS", "intent": "diet_form"}

    # A query that explicitly asks for a recipe wants recipes, not a tangential
    # definition/fact card that happens to score well on shared vocabulary (e.g.
    # "vitamin c recipes" pulling in the "Vitamin C" fact sheet itself) — "recipi"
    # also catches the common misspelling "recipies". Cast a wider net (top_k=15
    # instead of 5) so purpose-built recipes aren't crowded out of the candidate
    # pool by the much larger pool of generic, uncurated recipes.
    is_recipe_request = intent == "advisory" and any(
        k in query.lower() for k in
        ("recipe", "recipi", "milkshake", "smoothie", "how to make", "how to cook")
    )
    # A recipe genuinely rich in a named nutrient (e.g. vitamin B) doesn't
    # necessarily rank in the top ~15 by semantic similarity — "vitamin b" as a
    # query doesn't embed meaningfully differently from "vitamin c" or "vitamin
    # a", so the actually-relevant recipes can sit far down the ranking. Cast a
    # much wider net when a specific nutrient is named so the strict content
    # filter below has real candidates to work with, not just whatever the
    # semantic ranking happened to prefer.
    recipe_nutrient_terms = (
        [term for term in NUTRIENT_QUERY_TERMS if term in query.lower()] if is_recipe_request else []
    )
    retrieval_query = normalize_query_spelling(query)

    try:
        index, _ = get_bm25_index()
        vector_index = get_vector_index()
        if intent == "factual":
            top_k = 20
        elif recipe_nutrient_terms:
            top_k = 300
        elif is_recipe_request:
            top_k = 15
        else:
            top_k = 5
        search_results = hybrid_search(retrieval_query, index, vector_index, top_k=top_k)
    except Exception as e:
        return {
            "status": "SUCCESS", "intent": intent, "results": [], "rejected_results": [],
            "error": f"{type(e).__name__}: {e}",
        }

    if is_recipe_request:
        search_results = [
            r for r in search_results
            if index.doc_metadata.get(r["doc_id"], {}).get("category") == "Recipes"
        ]
        # A recipe can rank well for "vitamin b" purely on semantic similarity to
        # generic "vitamin X: ... (notably rich)" phrasing even when it has zero
        # actual vitamin B content — the fix is a literal, lexical requirement
        # that the specific nutrient be named in the recipe's content, not just
        # "vitamin-shaped" content in general.
        if recipe_nutrient_terms:
            nutrient_filtered = [
                r for r in search_results
                if any(
                    term in index.doc_metadata.get(r["doc_id"], {}).get("content", "").lower()
                    for term in recipe_nutrient_terms
                )
            ]
            if nutrient_filtered:
                search_results = nutrient_filtered

    if intent == "factual":
        # A recipe is never a valid answer to a definitional "what is X" query —
        # it's a dish, not information about the food itself — so skip past any
        # recipe results to find the best genuine factual/reference document.
        # Re-rank by raw semantic similarity rather than the BM25-heavy combined
        # score: a lexically-overlapping-but-off-topic doc (e.g. "Scurvy" for a
        # vitamin C query) can outrank the true best answer on keyword overlap
        # alone, even though the true answer is more semantically relevant.
        non_recipe = [
            r for r in search_results
            if index.doc_metadata.get(r["doc_id"], {}).get("category") != "Recipes"
        ]
        non_recipe.sort(key=lambda r: r["vector_score"], reverse=True)
        comparison = is_comparison_query(query)
        # A comparison query ("egg or chicken", "milk vs paneer") is asking about two
        # things at once — truncating to a single top result silently drops whichever
        # side scored lower, answering as though only one item was ever named.
        factual_results = non_recipe[:2] if comparison else non_recipe[:1]

        # If the query directly names a specific vitamin/mineral, that document
        # wins outright over whatever the generic vector-score ranking preferred —
        # semantic similarity alone is unreliable for disambiguating "vitamin a
        # deficiency" from the broader Vitamin Deficiency Overview or unrelated
        # docs that just happen to share more vocabulary. Skipped for comparison
        # queries, which already keep multiple results instead of collapsing to one.
        if not comparison:
            named_doc_id = (
                find_named_vitamin_mineral_doc(query, index)
                or find_named_diet_doc(query)
                or find_symptom_matched_vitamin_doc(query)
            )
            if named_doc_id:
                named_match = next((r for r in non_recipe if r["doc_id"] == named_doc_id), None)
                if named_match is None:
                    # A symptom-matched (or named) doc may not have surfaced in the
                    # top-k semantic candidate pool at all — its own page can dilute
                    # semantic similarity with unrelated content (RDA, food sources)
                    # while a generic "Deficiency Overview" doc's narrower vocabulary
                    # scores higher purely on overlap. Once matched with confidence,
                    # answer with it directly rather than silently falling through.
                    meta = index.doc_metadata.get(named_doc_id, {})
                    if meta:
                        named_match = {
                            "doc_id": named_doc_id,
                            "title": meta.get("title", ""),
                            "content": meta.get("content", ""),
                            "bm25_score": 0.0,
                            "vector_score": 1.0,
                        }
                if named_match:
                    factual_results = [named_match]

        # A "vitamin/mineral deficiency" query is really asking about the deficiency
        # DISEASE it causes (e.g. vitamin D deficiency -> Rickets), not just the
        # nutrient's general info page. Use a direct, known mapping rather than
        # picking whichever Deficiency Diseases doc happened to score highest by
        # vector similarity — that previously surfaced wrong pairings (e.g. Vitamin A
        # + Beriberi, or Vitamin B12 + Night Blindness) since generic "deficiency"
        # vocabulary overlap doesn't track which disease a given nutrient actually causes.
        if factual_results and "deficien" in query.lower():
            top_doc_id = factual_results[0]["doc_id"]
            top_category = index.doc_metadata.get(top_doc_id, {}).get("category")
            if top_category == "Vitamins & Minerals":
                linked_disease_id = VITAMIN_TO_DEFICIENCY_DISEASE.get(top_doc_id)
                linked_meta = index.doc_metadata.get(linked_disease_id, {}) if linked_disease_id else {}
                if linked_meta:
                    factual_results.append({
                        "doc_id": linked_disease_id,
                        "title": linked_meta.get("title", ""),
                        "content": linked_meta.get("content", ""),
                        "bm25_score": 0.0,
                        "vector_score": 1.0,
                    })

        return {"status": "SUCCESS", "intent": "factual", "results": factual_results, "rejected_results": []}

    scored = rank_by_trust(score_documents(retrieval_query, search_results, index))
    passed, rejected = apply_guardrail(scored, threshold=ADVISORY_TRUST_THRESHOLD)
    return {"status": "SUCCESS", "intent": "advisory", "results": passed, "rejected_results": rejected}
