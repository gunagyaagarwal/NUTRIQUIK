import re
import pandas as pd
import numpy as np

# Sample Curated Corpus for Immunonutrition & Health
SAMPLE_CORPUS = [
    {
        "id": "DOC001",
        "title": "Role of Vitamin D in T-cell Regulation and Respiratory Immunity",
        "source": "Journal of Clinical Immunology & WHO Guidelines",
        "source_credibility": 0.95,
        "category": "Immunology",
        "evidence_level": "Level I (Meta-Analysis / RCT)",
        "evidence_score": 0.92,
        "publication_year": 2024,
        "has_disclaimer": True,
        "text": "Vitamin D plays a critical role in promoting anti-inflammatory cytokine production (IL-10) while modulating T-helper cell responses. Supplementation reduces susceptibility to upper respiratory tract infections in serum-deficient populations.",
        "bm25_score": 0.88,
        "dense_score": 0.91,
        "safety_score": 0.98,
        "type": "Advisory"
    },
    {
        "id": "DOC002",
        "title": "Zinc Synergy with Vitamin C in Innate Pathogen Defense",
        "source": "Nutritional Reviews / USDA Database",
        "source_credibility": 0.90,
        "category": "Micronutrients",
        "evidence_level": "Level I (RCT)",
        "evidence_score": 0.88,
        "publication_year": 2023,
        "has_disclaimer": True,
        "text": "Zinc ions inhibit viral RNA polymerase activity, while Vitamin C enhances macrophage phagocytosis and leukocyte migration. Combined administration during early symptom onset shortens common cold duration by up to 28%.",
        "bm25_score": 0.84,
        "dense_score": 0.86,
        "safety_score": 0.95,
        "type": "Advisory"
    },
    {
        "id": "DOC003",
        "title": "Omega-3 Polyunsaturated Fatty Acids (EPA/DHA) and Inflammatory Markers",
        "source": "American Journal of Clinical Nutrition",
        "source_credibility": 0.92,
        "category": "Lipids & Inflammation",
        "evidence_level": "Level II (Cohort Study)",
        "evidence_score": 0.81,
        "publication_year": 2023,
        "has_disclaimer": True,
        "text": "High-dose EPA/DHA suppresses NF-kB pathway activation and lowers serum CRP and IL-6 levels in patients with chronic systemic inflammatory conditions.",
        "bm25_score": 0.79,
        "dense_score": 0.83,
        "safety_score": 0.94,
        "type": "Advisory"
    },
    {
        "id": "DOC004",
        "title": "Probiotic Strains (Lactobacillus & Bifidobacterium) in Mucosal Immunity",
        "source": "Frontiers in Cellular Microbiology",
        "source_credibility": 0.87,
        "category": "Gut Microbiome",
        "evidence_level": "Level II (Clinical Trial)",
        "evidence_score": 0.76,
        "publication_year": 2022,
        "has_disclaimer": True,
        "text": "Targeted probiotic strain supplementation strengthens gut epithelial tight junctions and increases secretory IgA synthesis in gut-associated lymphoid tissue (GALT).",
        "bm25_score": 0.75,
        "dense_score": 0.79,
        "safety_score": 0.90,
        "type": "Advisory"
    },
    {
        "id": "DOC005",
        "title": "Definition and Recommended Daily Intake of Vitamin C",
        "source": "USDA FoodData Central / Wikipedia Medical",
        "source_credibility": 0.98,
        "category": "Factual Reference",
        "evidence_level": "Level I (Official Standard)",
        "evidence_score": 0.95,
        "publication_year": 2025,
        "has_disclaimer": True,
        "text": "Vitamin C (ascorbic acid) is an essential water-soluble vitamin. Recommended Daily Allowance (RDA) for adult males is 90 mg/day and for adult females is 75 mg/day.",
        "bm25_score": 0.95,
        "dense_score": 0.94,
        "safety_score": 1.00,
        "type": "Factual"
    },
    {
        "id": "DOC006",
        "title": "Anecdotal Herbal Cure for Autoimmune Diseases via Unverified Extract X",
        "source": "Unverified Alternative Wellness Blog",
        "source_credibility": 0.25,
        "category": "Herbal Supplements",
        "evidence_level": "Level IV (Anecdotal Blog)",
        "evidence_score": 0.20,
        "publication_year": 2019,
        "has_disclaimer": False,
        "text": "Drinking megadoses of Extract X twice daily completely reverses rheumatoid arthritis without any medical supervision or conventional medication.",
        "bm25_score": 0.62,
        "dense_score": 0.45,
        "safety_score": 0.15,
        "type": "Advisory"
    },
    {
        "id": "DOC007",
        "title": "High-Dose Vitamin K2 Interaction with Anticoagulants (Warfarin)",
        "source": "Clinical Toxicology Journal",
        "source_credibility": 0.94,
        "category": "Medication Contraindications",
        "evidence_level": "Level I (Clinical Trial)",
        "evidence_score": 0.90,
        "publication_year": 2024,
        "has_disclaimer": True,
        "text": "Vitamin K directly antagonizes the therapeutic effect of Vitamin K Antagonists (VKAs) such as Warfarin, leading to altered INR and severe thrombotic risk.",
        "bm25_score": 0.81,
        "dense_score": 0.85,
        "safety_score": 0.99,
        "type": "Advisory"
    },
    {
        "id": "DOC008",
        "title": "Miracle Bleach Detox and Immuno-Cleanse Therapy",
        "source": "Banned Health Forum",
        "source_credibility": 0.05,
        "category": "Harmful Protocol",
        "evidence_level": "Level IV (Fake News)",
        "evidence_score": 0.05,
        "publication_year": 2018,
        "has_disclaimer": False,
        "text": "Ingesting chemical disinfectants flushes toxins and eliminates all pathogens instantly.",
        "bm25_score": 0.50,
        "dense_score": 0.30,
        "safety_score": 0.00,
        "type": "Advisory"
    }
]

# Query Guard Keyword Blocklist & Safety Checks
BLOCKED_KEYWORDS = ["poison", "kill", "weapons", "violence", "self-harm", "bleach", "suicide", "bomb", "cyanide", "illegal drugs"]

def check_query_guard(query: str):
    """
    Query Guard Module: ML Logistic Regression + Keyword blocklist check.
    Returns (is_safe: bool, reason: str)
    """
    query_lower = query.lower().strip()
    
    # Keyword blocklist check
    for kw in BLOCKED_KEYWORDS:
        if kw in query_lower:
            return False, f"Blocked by Query Guard: High risk content detected ('{kw}'). Requests involving hazardous substances, self-harm, or illegal procedures are prohibited."
            
    # Length / Off-domain check
    if len(query_lower) < 3:
        return False, "Query Guard: Query too short or ambiguous."
        
    # Simulated ML LogReg probability score
    off_domain_terms = ["crypto", "stock market", "car repair", "gaming", "python code"]
    for od in off_domain_terms:
        if od in query_lower:
            return False, f"Query Guard: Out-of-domain query detected ('{od}'). NUTRIQUIK is domain-constrained to nutrition and immunology."
            
    return True, "Safe query passed through Query Guard."

def route_intent(query: str):
    """
    Intent Router Module: Classifies query into 'Factual' vs 'Advisory'
    """
    query_lower = query.lower()
    factual_triggers = ["what is", "definition", "rda of", "recommended daily", "chemical formula", "who defined", "how many mg"]
    
    for tr in factual_triggers:
        if tr in query_lower:
            return "Factual"
            
    return "Advisory"

def calculate_trust_score(doc):
    """
    Calculates Composite Trust Score using project report formula:
    Trust Score = (0.5 * Relevance) + (0.3 * Evidence Score) + (0.2 * Safety Score)
    where Relevance = 0.5 * BM25 + 0.5 * Dense Cosine
    """
    relevance = (doc["bm25_score"] + doc["dense_score"]) / 2.0
    evidence = doc["evidence_score"]
    safety = doc["safety_score"]
    
    trust_score = (0.5 * relevance) + (0.3 * evidence) + (0.2 * safety)
    return round(trust_score, 3), round(relevance, 3)

def run_retrieval_pipeline(query: str):
    """
    Executes full pipeline: Query Guard -> Intent Router -> Retrieval & XGBoost Scoring -> Guardrail Filter
    """
    is_safe, guard_reason = check_query_guard(query)
    if not is_safe:
        return {
            "status": "REJECTED_BY_GUARD",
            "reason": guard_reason,
            "intent": "Unknown",
            "results": [],
            "rejected_results": []
        }
        
    intent = route_intent(query)
    processed_results = []
    
    # Calculate score for each doc
    for doc in SAMPLE_CORPUS:
        # Simple term matching modulation for interactivity
        query_words = set(re.findall(r'\w+', query.lower()))
        doc_words = set(re.findall(r'\w+', (doc["title"] + " " + doc["text"]).lower()))
        overlap = len(query_words.intersection(doc_words))
        
        # Adjust BM25/Dense dynamically based on query match
        boost = min(0.3, overlap * 0.08)
        
        doc_copy = doc.copy()
        doc_copy["bm25_score"] = min(0.99, doc["bm25_score"] + boost)
        doc_copy["dense_score"] = min(0.99, doc["dense_score"] + boost)
        
        trust_score, relevance_score = calculate_trust_score(doc_copy)
        doc_copy["trust_score"] = trust_score
        doc_copy["relevance_score"] = relevance_score
        
        # Guardrail Filter Rejection logic (Trust < 0.50 OR Safety == 0)
        rejection_reasons = []
        if trust_score < 0.50:
            rejection_reasons.append("Low Trust Score (< 50%)")
        if doc_copy["safety_score"] < 0.30:
            rejection_reasons.append("Safety Violation / Harmful Content")
        if not doc_copy["has_disclaimer"] and trust_score < 0.60:
            rejection_reasons.append("Missing Medical Disclaimer")
        if doc_copy["evidence_score"] < 0.40:
            rejection_reasons.append("Low Evidence Level (< 0.40)")
            
        doc_copy["is_rejected"] = len(rejection_reasons) > 0
        doc_copy["rejection_reasons"] = rejection_reasons
        
        processed_results.append(doc_copy)
        
    # Sort by Trust Score descending
    processed_results.sort(key=lambda x: x["trust_score"], reverse=True)
    
    accepted = [d for d in processed_results if not d["is_rejected"]]
    rejected = [d for d in processed_results if d["is_rejected"]]
    
    return {
        "status": "SUCCESS",
        "reason": guard_reason,
        "intent": intent,
        "results": accepted,
        "rejected_results": rejected,
        "all_results": processed_results
    }

def get_shap_values(doc):
    """
    Simulates SHAP feature explanations for XGBoost Trust Scoring
    """
    bm25_contrib = (doc["bm25_score"] - 0.5) * 0.25
    dense_contrib = (doc["dense_score"] - 0.5) * 0.25
    evidence_contrib = (doc["evidence_score"] - 0.5) * 0.30
    source_contrib = (doc["source_credibility"] - 0.5) * 0.15
    safety_contrib = (doc["safety_score"] - 0.5) * 0.20
    disclaimer_contrib = 0.05 if doc["has_disclaimer"] else -0.15
    
    return {
        "BM25 Lexical Score": round(bm25_contrib, 3),
        "Dense Embedding Cosine": round(dense_contrib, 3),
        "Evidential Quality": round(evidence_contrib, 3),
        "Source Credibility": round(source_contrib, 3),
        "Safety Rating": round(safety_contrib, 3),
        "Disclaimer Presence": round(disclaimer_contrib, 3)
    }

def get_interaction_matrix():
    """
    Returns nutrient-medication-condition contraindications and synergies for profile analysis
    """
    return [
        {"nutrient": "Vitamin D3", "target": "T-Cell Immunity", "type": "Synergy", "detail": "Promotes anti-inflammatory immune tolerance"},
        {"nutrient": "Zinc", "target": "Viral Polymerase", "type": "Inhibition", "detail": "Inhibits viral replication when taken within 24h of symptoms"},
        {"nutrient": "Vitamin K2", "target": "Warfarin / Anticoagulants", "type": "Contraindication", "detail": "High risk: Antagonizes Warfarin activity, alters INR"},
        {"nutrient": "High-dose Iron", "target": "Zinc Absorption", "type": "Competition", "detail": "Competes for intestinal DMT1 transporters"},
        {"nutrient": "Omega-3 (EPA/DHA)", "target": "NF-kB Pathway", "type": "Synergy", "detail": "Downregulates IL-6 and TNF-alpha expression"},
        {"nutrient": "Vitamin C", "target": "Iron Absorption", "type": "Synergy", "detail": "Converts ferric (Fe3+) to ferrous (Fe2+) for enhanced uptake"}
    ]
