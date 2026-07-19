import os
import re

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "domain_classifier.pkl")

DOMAIN_CONFIDENCE_THRESHOLD = 0.5

BLOCKLIST_TERMS = [
    "poison", "kill", "murder", "suicide", "self-harm", "self harm",
    "drug abuse", "overdose", "weapon", "bomb", "explosive", "attack",
    "torture", "harm myself", "hurt someone", "lethal", "how to make drugs",
]

BLOCKLIST_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in sorted(BLOCKLIST_TERMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

HARMFUL_CONTENT_RESULT = {
    "allowed": False,
    "reason": "harmful_content",
    "message": "I can't help with harmful content. I only answer nutrition and health questions.",
    "suggestion": "Try: 'What are the benefits of Vitamin C?'",
}

IN_DOMAIN_EXAMPLES = [
    # Core nutrition/vitamin/mineral facts
    "what is vitamin c", "benefits of zinc", "diet plan for diabetes", "how does immunity work",
    "foods rich in iron", "symptoms of anemia", "nutrition for heart health", "what causes scurvy",
    "best foods for weight loss", "how to boost immunity", "vitamin d deficiency symptoms",
    "protein sources for vegetarians", "calorie intake for weight gain", "liver disease prevention diet",
    "thyroid diet plan", "calcium rich foods", "what is BMI", "iron deficiency effects",
    "healthy breakfast options", "omega 3 benefits", "antioxidant rich foods", "gut health and immunity",
    "zinc supplement dosage", "vitamin b12 sources", "fiber rich foods", "low sugar diet plan",
    "anti inflammatory foods list", "kidney friendly diet", "cholesterol lowering foods",
    "magnesium deficiency", "probiotic foods benefits", "folic acid importance", "superfoods for immunity",
    "turmeric health benefits", "immune system boosters", "diet for PCOS", "hypothyroid diet foods",
    "weight management tips", "sports nutrition basics", "creatine supplement benefits",
    "predict my diabetes risk", "assess my anemia risk", "check heart disease risk",
    # Diabetes / blood sugar phrasing variety
    "diabetic diet plan", "foods to avoid if diabetic", "what can diabetics eat",
    "best diet for type 2 diabetes", "is rice good for diabetics", "sugar free diet for diabetes",
    "low carb diet for diabetics", "blood sugar control diet", "foods that lower blood sugar",
    "insulin resistance diet", "what foods should a diabetic avoid", "can diabetics eat fruit",
    "is banana good for diabetics", "how to manage diabetes with diet", "diabetic friendly snacks",
    # Anemia / iron
    "iron rich foods for anemia", "anemia diet plan", "foods that help with anemia",
    "vegetarian iron sources", "low hemoglobin diet", "signs of iron deficiency anemia",
    "how to increase hemoglobin naturally", "best foods for low iron",
    # Heart / cholesterol / blood pressure
    "heart healthy diet", "foods good for heart health", "cholesterol lowering diet plan",
    "low sodium diet for blood pressure", "foods to avoid for high blood pressure",
    "omega 3 for heart health", "best diet for cardiac patients", "foods that raise cholesterol",
    "diet to reduce blood pressure naturally",
    # Liver
    "liver detox diet", "foods good for liver health", "fatty liver disease diet",
    "foods to avoid for liver damage", "diet for liver cirrhosis",
    # Kidney
    "kidney disease diet plan", "foods to avoid with kidney disease",
    "low potassium diet for kidney patients", "renal diet foods", "diet for kidney stones",
    # Thyroid
    "hypothyroidism diet plan", "foods good for thyroid health",
    "foods to avoid with thyroid problems", "hyperthyroidism diet", "iodine and thyroid function",
    # Vitamins / minerals deep dive
    "vitamin a benefits", "vitamin b12 deficiency symptoms", "vitamin d rich foods",
    "vitamin e benefits for skin", "vitamin k foods", "magnesium rich foods", "potassium rich foods",
    "calcium sources for vegetarians", "zinc deficiency symptoms", "selenium benefits",
    "iodine rich foods", "folic acid rich foods", "signs of vitamin deficiency",
    "which vitamin deficiency causes hair loss", "multivitamin dosage guide",
    # Weight management
    "how to lose weight fast", "diet plan for weight gain", "calorie deficit diet",
    "best exercises for weight loss diet", "high protein diet for muscle gain",
    "healthy snacks for weight loss", "is intermittent fasting good for weight loss",
    # Immunity / immunology
    "foods that boost immune system", "how to strengthen immunity naturally",
    "vitamin c for immunity", "immune boosting supplements", "how does the immune system work",
    "role of antibodies in immunity", "t cells and immune response", "innate vs adaptive immunity",
    "how nutrition affects immune function", "foods that weaken the immune system",
    # Supplements
    "creatine benefits for muscle", "protein powder side effects", "multivitamin benefits",
    "fish oil supplement benefits", "probiotic supplement benefits", "best supplements for athletes",
    "is it safe to take supplements daily", "whey protein vs plant protein",
    # Diet types
    "vegan diet benefits", "keto diet plan", "mediterranean diet benefits",
    "intermittent fasting benefits", "gluten free diet benefits", "paleo diet foods",
    "balanced diet meal plan", "low fodmap diet foods",
    # General nutrition science
    "macronutrients and micronutrients", "how many calories should i eat",
    "high fiber foods list", "antioxidant rich fruits", "anti inflammatory diet foods",
    "gut health foods", "probiotics vs prebiotics", "healthy fats vs unhealthy fats",
    "protein requirement per day", "signs of malnutrition", "foods rich in antioxidants",
    "best breakfast for energy", "hydration and health benefits", "what nutrients does the body need",
    "difference between complex and simple carbs",
    # Prediction-style health risk phrasing
    "predict my heart disease risk", "check my kidney function risk", "assess my thyroid risk",
    "am i at risk of diabetes", "screen for anemia risk", "evaluate my liver health risk",
    "what is my risk of vitamin deficiency",
    # PCOS / hormonal
    "pcos diet plan", "foods for pcos patients", "hormonal imbalance diet",
    "foods to avoid with pcos",
    # Digestion / common remedies
    "natural ways to cure acidity", "foods that cause bloating", "home remedies for acid reflux",
    "how to improve digestion naturally", "foods good for gut health",
    # Natural casual phrasing: immune system basics
    "what are the two parts of the immune system", "innate vs adaptive immunity", "what is innate immunity",
    "what do helper t cells do", "cd4 cells function", "what kills virus infected cells",
    "what do killer t cells do", "what makes antibodies", "types of antibodies",
    "which antibody crosses the placenta", "what does hiv attack",
    # Natural casual phrasing: vitamins & minerals
    "vitamin c benefits", "why do i need vitamin c", "what is vitamin d for", "sunshine vitamin",
    "what does zinc do", "why is zinc important", "what is iron good for", "why do i need iron",
    "where to get b12", "which vitamins are antioxidants", "vitamins for bones",
    "what's good for bone health", "what is folate", "folic acid in pregnancy",
    "what does calcium do", "what is iodine for", "iodine and thyroid",
    # Natural casual phrasing: deficiency diseases
    "what is rickets", "soft bones in kids", "what causes goiter", "swollen thyroid",
    "what is beriberi", "what is pellagra", "the 4 ds", "what is night blindness",
    # Natural casual phrasing: superfoods
    "is turmeric good for you", "is ginger good for nausea", "is garlic good for you",
    "garlic for blood pressure", "is green tea healthy", "are berries healthy",
    "blueberry benefits", "are fermented foods good",
    # Natural casual phrasing: diet types
    "what is the keto diet", "explain keto", "what is the mediterranean diet",
    "what is a vegan diet", "vegan vs vegetarian",
    # Natural casual phrasing: advisory
    "foods to strengthen immune system", "best iron rich foods", "how to raise my iron",
    "foods for bone health", "how to improve bone strength", "what should i eat for high cholesterol",
    "foods that fight fatigue", "why am i always tired what should i eat", "best foods for gut health",
    "what helps digestion", "anti-inflammatory foods", "what should i eat to reduce inflammation",
    "supplement for muscle recovery", "best supplement after workout", "what helps muscle soreness",
    "foods for skin health", "how to boost my immune system naturally", "what should a diabetic eat",
    "meal plan to lose weight", "what to eat for hypertension", "what should i eat for hypothyroidism",
    # Natural casual phrasing: prediction
    "am i at risk of diabetes", "do i have diabetes", "am i anemic", "check if i have anemia",
    "heart health check", "am i at risk of kidney disease", "check my kidney health",
    "do i have a thyroid problem", "am i at risk of liver disease", "which deficiency do i have",
    "recommend a supplement for me",
    # Fruits & common whole foods
    "is kiwi healthy", "kiwi nutrition facts", "benefits of eating kiwi", "show me the benefits of kiwi",
    "is mango good for you", "mango health benefits", "is banana healthy", "benefits of eating banana",
    "is apple good for you", "nutritional value of apple", "benefits of eating oranges",
    "is avocado healthy", "avocado nutrition facts", "benefits of eating berries",
    "is watermelon good for hydration", "nutritional value of spinach", "benefits of eating broccoli",
    "is pineapple healthy", "benefits of citrus fruits", "healthiest fruits to eat",
    # Common infections/conditions and lab-report shorthand (nutrition/immune support
    # during illness and recovery, and interpreting basic blood work, are in-domain)
    "covid 19", "what is covid", "diet during covid recovery", "foods to eat during covid",
    "immunity boosting foods for covid", "what is malaria", "diet during malaria recovery",
    "foods to eat during malaria", "what is dengue", "dengue fever symptoms",
    "foods to eat during dengue fever", "platelet count in dengue", "what is chikungunya",
    "chikungunya symptoms", "recovery diet for chikungunya", "what is asthma",
    "asthma diet", "foods that trigger asthma", "asthma and nutrition",
    "what is haemoglobin", "haemoglobin levels", "low haemoglobin symptoms",
    "how to increase haemoglobin", "what is a cbc report", "cbc report", "cbc test results",
    "understanding my cbc blood test",
    # More common illnesses (nutrition/immune support during illness and recovery)
    "what is tuberculosis", "tb symptoms", "diet for tb patients", "what is coronavirus",
    "diet during coronavirus recovery", "what is chicken pox", "chickenpox diet",
    "foods to eat during chickenpox", "what is jaundice", "diet for jaundice patients",
    "foods to avoid with jaundice", "what is food poisoning", "food poisoning diet",
    "what to eat after food poisoning", "what is hepatitis", "diet for hepatitis patients",
    "liver friendly foods for hepatitis", "what is cholera", "cholera treatment diet",
    "what is diarrhea", "diarrhea diet", "foods to eat during diarrhea",
    "what is depression", "foods that help with depression", "nutrition for mental health",
    "what is typhoid", "typhoid diet", "foods to eat during typhoid fever",
    "what is the flu", "flu diet", "foods to eat when you have the flu",
    "immunity boosting foods for flu recovery",
    # HIV/AIDS and cancer (sources, prevention, and supportive nutrition)
    "what is hiv", "how is hiv transmitted", "hiv prevention", "diet for hiv patients",
    "nutrition for people living with hiv", "what is aids", "difference between hiv and aids",
    "aids prevention", "diet for aids patients", "what is cancer", "causes of cancer",
    "cancer prevention", "foods that help prevent cancer", "diet during cancer treatment",
    "nutrition for cancer patients", "antioxidant foods for cancer prevention",
    # Cancer types
    "what is breast cancer", "what is lung cancer", "what is skin cancer", "what is leukemia",
    "what is colorectal cancer", "what is prostate cancer", "what is cervical cancer",
    "what is liver cancer", "what is stomach cancer", "what is oral cancer",
    "what is lymphoma", "what is pancreatic cancer",
    # Fruits and vegetables
    "is apple healthy", "banana nutrition facts", "orange health benefits",
    "is cucumber healthy", "cauliflower nutrition", "cabbage health benefits",
    "bell pepper nutrition", "capsicum health benefits", "sweet potato nutrition",
    "beetroot health benefits", "benefits of beets",
    # Milkshakes/smoothies
    "milkshake recipes", "protein milkshake", "healthy milkshake", "chocolate milkshake recipe",
    "banana milkshake benefits", "smoothie recipes", "healthy smoothie for breakfast",
]

OUT_OF_DOMAIN_EXAMPLES = [
    "how to fix my car", "write me a poem", "stock market tips", "best movies 2024",
    "python programming tutorial", "how to cook pasta", "weather tomorrow", "translate to french",
    "solve this math equation", "who is the president", "play a song", "tell me a joke",
    "book a flight", "football scores today", "how to paint a wall", "fix my laptop screen",
    "history of ancient rome", "space exploration news", "cryptocurrency trading", "fashion trends 2025",
    "car insurance quote", "how to swim faster", "guitar chords for beginners", "photography tips",
    "gardening basics", "dog training commands", "real estate prices", "travel to japan guide",
    "chess opening strategy", "makeup tutorial", "best video games", "how to code in java",
    "recipe for pasta", "best smartphone 2025", "learn spanish online",
    # Tech / programming
    "latest smartphone reviews", "best programming languages 2025", "how to build a website",
    "web development frameworks", "database design tips", "cloud computing basics",
    "cybersecurity tips", "how to fix wifi issues", "electric car reviews",
    # Finance / business
    "how to invest in stocks", "how to start a business", "tax filing deadline",
    "mortgage rates today", "credit score improvement", "how to save money",
    "how to write a resume", "job interview tips",
    # Travel / lifestyle
    "tourist places in paris", "best travel destinations 2025", "flight booking tips",
    "hotel recommendations", "visa application process",
    # Pets / animals
    "how to train a puppy", "best dog breeds", "cat behavior tips",
    # Science / general knowledge (non-health)
    "history of world war 2", "solar system planets", "how black holes form",
    "chemistry periodic table", "algebra homework help", "geometry problems",
    "airplane mechanics", "rocket science basics", "how engines work",
    # Entertainment / sports / news
    "political news update", "election results", "sports news today", "football match highlights",
    "basketball rules", "tennis tournament schedule", "top 10 songs this week",
    # Arts / hobbies
    "how to knit a scarf", "sewing patterns", "art history facts", "famous paintings",
    "music theory basics", "learn to play piano", "how to learn guitar", "best laptops for gaming",
    "car maintenance tips",
]

OFF_DOMAIN_RESULT = {
    "allowed": False,
    "reason": "off_domain",
    "message": "I can only help with nutrition and health-related questions.",
    "suggestion": "Try: 'What foods are rich in iron?'",
}

# Substring stems (not whole words) so variants like "diabetic"/"diabetes" or
# "deficiency"/"deficient" all match. Used only as a fallback when the domain
# classifier has zero training-vocabulary overlap with the query and therefore
# no real signal to act on.
DOMAIN_KEYWORDS = [
    "nutrition", "diet", "food", "eat", "meal", "vitamin", "mineral", "protein",
    "calorie", "immun", "supplement", "health", "disease", "symptom", "deficien",
    "diabet", "anemia", "hemoglobin", "heart", "cardiac", "cholesterol", "liver",
    "kidney", "renal", "thyroid", "obesity", "weight", "bmi", "fiber", "carb",
    "fat", "sugar", "glucose", "iron", "calcium", "zinc", "omega", "antioxidant",
    "probiotic", "allergy", "recipe", "cook", "acid", "digest", "gut", "cure",
    "remedy", "bloat", "hormon", "pcos", "platelet", "electrolyte", "gout",
    "bone", "hair loss", "pregnan", "blood pressure", "blood sugar",
    # Fruits, vegetables & common whole foods (so "is kiwi healthy" etc. aren't
    # blocked just because they share no vocabulary with the training examples)
    "fruit", "vegetable", "kiwi", "mango", "banana", "apple", "orange", "grape",
    "papaya", "pineapple", "guava", "pomegranate", "watermelon", "strawberry",
    "blueberry", "raspberry", "melon", "pear", "plum", "peach", "apricot", "fig",
    "lemon", "lime", "avocado", "coconut", "spinach", "broccoli", "carrot",
    "tomato", "potato", "onion", "garlic", "berries", "citrus",
    # Common infections/conditions relevant to immunonutrition (nutrition/immune
    # support during illness, recovery diets, blood work interpretation), plus the
    # British spelling of hemoglobin and common lab-report shorthand.
    "covid", "malaria", "dengue", "chikungunya", "asthma", "haemoglobin", "cbc",
    # Common misspellings of chikungunya seen in real queries.
    "chickengunia", "chickungunya", "chickungunia",
    # More common illnesses.
    "tuberculosis", "coronavirus", "chickenpox", "chicken pox", "jaundice",
    "food poisoning", "hepatitis", "cholera", "diarrhea", "diarrhoea",
    "depression", "typhoid", "influenza", "flu",
    "hiv", "aids", "cancer", "tumor", "tumour", "oncology", "leukemia", "leukaemia", "lymphoma",
    # More vegetables not already covered above.
    "cucumber", "cauliflower", "cabbage", "bell pepper", "capsicum", "sweet potato", "beetroot", "beet",
    "milkshake", "smoothie",
]


def check_blocklist(query):
    return bool(BLOCKLIST_PATTERN.search(query))


def has_domain_keyword(query):
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in DOMAIN_KEYWORDS)


def train_domain_classifier():
    texts = IN_DOMAIN_EXAMPLES + OUT_OF_DOMAIN_EXAMPLES
    labels = [1] * len(IN_DOMAIN_EXAMPLES) + [0] * len(OUT_OF_DOMAIN_EXAMPLES)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        # class_weight="balanced" matters here: IN_DOMAIN_EXAMPLES keeps growing faster
        # than OUT_OF_DOMAIN_EXAMPLES as new in-domain vocabulary gets added, and an
        # unweighted LogisticRegression drifts its decision boundary toward "allow"
        # as that imbalance grows, causing previously-correct off-domain rejections
        # (e.g. "what's the weather", "best laptop to buy") to start passing.
        ("clf", LogisticRegression(class_weight="balanced")),
    ])
    pipeline.fit(texts, labels)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


def load_domain_classifier():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train_domain_classifier()


def classify_domain(query, model):
    tfidf = model.named_steps["tfidf"]
    vec = tfidf.transform([query])
    if vec.nnz == 0:
        return 0.65 if has_domain_keyword(query) else 0.0

    proba = model.predict_proba([query])[0]
    classes = list(model.classes_)
    in_domain_confidence = proba[classes.index(1)]
    return in_domain_confidence


def run_query_guard(query, model=None):
    if check_blocklist(query):
        return dict(HARMFUL_CONTENT_RESULT)

    if model is None:
        model = load_domain_classifier()

    confidence = classify_domain(query, model)
    if confidence < DOMAIN_CONFIDENCE_THRESHOLD:
        result = dict(OFF_DOMAIN_RESULT)
        result["confidence"] = float(confidence)
        return result

    return {
        "allowed": True,
        "reason": None,
        "confidence": float(confidence),
    }


if __name__ == "__main__":
    model = load_domain_classifier()

    test_queries = [
        "benefits of zinc",
        "how to make a bomb",
        "fix my car engine",
        "diet plan for immunity",
        "what's the weather",
        "predict diabetes risk",
        "best laptop to buy",
    ]

    for query in test_queries:
        result = run_query_guard(query, model=model)
        print(f"Query: '{query}'")
        print(f"  -> {result}")
