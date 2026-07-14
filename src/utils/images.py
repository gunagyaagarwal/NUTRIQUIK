import requests

TOPIC_MAPPING = {
    "vitamin c": "Vitamin_C", "zinc": "Zinc", "iron": "Iron", "calcium": "Calcium",
    "anemia": "Anemia", "diabetes": "Diabetes", "thyroid": "Thyroid", "immunity": "Immune_system",
    "turmeric": "Turmeric", "creatine": "Creatine", "protein": "Protein_(nutrient)",
    "cholesterol": "Cholesterol", "hemoglobin": "Hemoglobin", "obesity": "Obesity",
    "heart": "Heart", "liver": "Liver", "kidney": "Kidney",
    "vitamin d": "Vitamin_D", "vitamin b12": "Vitamin_B12", "vitamin a": "Vitamin_A",
    "vitamin e": "Vitamin_E", "vitamin k": "Vitamin_K", "magnesium": "Magnesium",
    "potassium": "Potassium", "fiber": "Dietary_fiber", "omega 3": "Omega-3_fatty_acid",
    "probiotics": "Probiotic", "antioxidants": "Antioxidant", "folate": "Folate",
    "selenium": "Selenium", "sodium": "Sodium", "fatty liver": "Fatty_liver_disease",
    "hypertension": "Hypertension", "pcos": "Polycystic_ovary_syndrome", "biotin": "Biotin",
    "collagen": "Collagen", "melatonin": "Melatonin", "blood sugar": "Blood_sugar_level",
}

WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
HEADERS = {"User-Agent": "NutriQuik/1.0 (educational project; contact@example.com)"}
_cache = {}


def get_topic_image(query):
    query_lower = query.lower()
    topic = next((v for k, v in TOPIC_MAPPING.items() if k in query_lower), None)
    if topic is None:
        return None

    if topic in _cache:
        return _cache[topic]

    try:
        resp = requests.get(WIKI_SUMMARY_URL.format(topic), headers=HEADERS, timeout=3)
        resp.raise_for_status()
        data = resp.json()
        url = data.get("thumbnail", {}).get("source")
    except Exception:
        url = None

    _cache[topic] = url
    return url


FALLBACK_COLORS = ["#4C6EF5", "#F76707", "#12B886", "#E64980", "#7048E8"]


def get_fallback_icon_html(topic):
    letter = topic[0].upper() if topic else "?"
    color = FALLBACK_COLORS[hash(topic) % len(FALLBACK_COLORS)]
    return (
        f'<div style="width:40px;height:40px;border-radius:50%;background:{color};'
        f'color:white;display:flex;align-items:center;justify-content:center;'
        f'font-weight:bold;font-size:18px;">{letter}</div>'
    )


if __name__ == "__main__":
    print(get_topic_image("zinc"))
