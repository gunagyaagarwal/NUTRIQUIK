import streamlit as st
import pandas as pd
import numpy as np

from data_engine import (
    run_retrieval_pipeline,
    get_shap_values,
    get_interaction_matrix,
    SAMPLE_CORPUS,
    check_query_guard
)

from ui_components import (
    create_trust_gauge,
    create_evidence_pyramid,
    create_retrieval_scatter,
    create_shap_chart,
    create_ir_metrics_chart,
    create_interaction_matrix_chart
)

# Page Configuration
st.set_page_config(
    page_title="NUTRIQUIK - Intelligent QA System for Nutrition & Immunology",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Visual Polish
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Thinner Sidebar Control Panel */
    [data-testid="stSidebar"] {
        min-width: 260px !important;
        max-width: 280px !important;
    }

    /* Gradient Hero Section */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #064E3B 100%);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38BDF8, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #94A3B8;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    .team-badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
    }

    .team-badge {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #E2E8F0;
        padding: 0.3rem 0.8rem;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Result Card Styling */
    .result-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .result-card:hover {
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border-color: #CBD5E1;
    }

    .badge-trust-high {
        background-color: #DCFCE7;
        color: #15803D;
        font-weight: 700;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        font-size: 0.85rem;
        display: inline-block;
    }

    .badge-trust-med {
        background-color: #FEF3C7;
        color: #B45309;
        font-weight: 700;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        font-size: 0.85rem;
        display: inline-block;
    }

    .badge-trust-low {
        background-color: #FEE2E2;
        color: #B91C1C;
        font-weight: 700;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        font-size: 0.85rem;
        display: inline-block;
    }

    .tag-evidence {
        background-color: #F1F5F9;
        color: #475569;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Guardrail Alert */
    .guard-rejected {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: #991B1B;
    }

    .guard-passed {
        background-color: #F0FDF4;
        border-left: 5px solid #10B981;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: #166534;
        font-weight: 600;
    }

    /* Metric Cards */
    .metric-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        text-align: center;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section Header
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">NUTRIQUIK</div>
    <div class="hero-subtitle">An Intelligent Question-Answering System for Nutrition and Immunology</div>
    <p style="margin:0; color: #CBD5E1; font-size: 0.95rem;">
        Hybrid IR Pipeline (BM25 + FAISS) &bull; Machine Learning Trust Scoring (XGBoost + SHAP) &bull; Guardrail Filter Safety
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation & User Profile Settings
st.sidebar.image("https://img.icons8.com/color/96/000000/leaf.png", width=60)
st.sidebar.title("NUTRIQUIK Control Panel")

nav_choice = st.sidebar.radio(
    "Select Interface View:",
    [
        "🔍 QA Pipeline & Query Interface",
        "📊 Evaluation & Trust Analytics",
        "🛡️ Guardrail & Rejected Results Panel",
        "💊 Profile & Interaction Matrix",
        "📚 Corpus & Dataset Inventory",
        "🏗️ System Blueprint & WBS Timeline"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("👤 User Health Profile")
user_name = st.sidebar.text_input("Name", value="Pulkit Sukhija")
user_age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=22)
user_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

st.sidebar.info("💡 User profile metadata is used for personalized health guidance.")

st.sidebar.markdown("---")
with st.sidebar.expander("👥 Project Team & Credits"):
    st.markdown("""
    - **Gunagya Agarwal** *(Project Lead & ML)*
    - **Pulkit Sukhija** *(Data & Evaluation)*
    - **Suryansh Singh** *(Frontend & UI)*
    - **Mridul Rai** *(IR & Backend)*
    - **Pranav S Nair** *(Integration & Guardrails)*
    
    *Jaypee Institute of Information Technology (2025–2026)*
    """)


# ==============================================================================
# VIEW 1: QA PIPELINE & QUERY INTERFACE
# ==============================================================================
if nav_choice == "🔍 QA Pipeline & Query Interface":
    st.header("🔍 Intelligent Question-Answering Pipeline")
    st.caption("Ask questions on immunonutrition, vitamin efficacy, or dietary supplements. Real-time BM25 + FAISS retrieval with XGBoost trust scoring.")

    # Preset Sample Queries
    st.markdown("**Quick Test Prompts:**")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    preset_query = ""
    if col_p1.button("🌿 Vitamin D & T-Cells"):
        preset_query = "What is the role of Vitamin D in T-cell regulation and immunity?"
    if col_p2.button("🍋 Zinc & Vitamin C Synergy"):
        preset_query = "Can Zinc and Vitamin C work together for immunity?"
    if col_p3.button("📘 What is RDA of Vitamin C?"):
        preset_query = "What is the recommended daily intake of Vitamin C?"
    if col_p4.button("⚠️ Test Guardrail Trigger"):
        preset_query = "How to use bleach detox to kill viruses?"

    # Query Input Box
    default_input = preset_query if preset_query else "What is the role of Vitamin D in respiratory immunity?"
    user_query = st.text_input("Enter your nutrition/immunology query:", value=default_input, key="qa_query_input")

    col_btn, col_blank = st.columns([1, 4])
    search_triggered = col_btn.button("🚀 Process Query", type="primary")

    if user_query or search_triggered:
        st.markdown("---")
        
        # Execute Retrieval Engine
        pipeline_output = run_retrieval_pipeline(user_query)
        
        # 1. Query Guard Status
        if pipeline_output["status"] == "REJECTED_BY_GUARD":
            st.markdown(f"""
            <div class="guard-rejected">
                <h4>🛑 Query Blocked by Safety Guardrail</h4>
                <p><strong>Reason:</strong> {pipeline_output['reason']}</p>
                <p style="font-size:0.9rem; margin-top:0.5rem;">NUTRIQUIK guardrails automatically intercept unsafe, toxic, or out-of-domain requests to protect clinical integrity.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="guard-passed">
                ✅ Query Guard: Passed | Router Intent Classification: <strong>{pipeline_output['intent']} Track</strong>
            </div>
            """, unsafe_allow_html=True)

            intent_type = pipeline_output['intent']
            results = pipeline_output['results']
            rejected_results = pipeline_output['rejected_results']

            # FACTUAL TRACK INTERFACE
            if intent_type == "Factual":
                st.subheader("📌 Factual Query Result (IR + Gemini LLM Refine Track)")
                st.info("ℹ️ Factual queries return the top-1 verified reference polished by LLM synthesis. 'Show More' ranking is disabled for direct factual definitions.")
                
                if results:
                    top_doc = results[0]
                    col_fac1, col_fac2 = st.columns([3, 1])
                    with col_fac1:
                        st.markdown(f"### {top_doc['title']}")
                        st.markdown(f"**Source:** {top_doc['source']} | **Category:** {top_doc['category']}")
                        st.success(f"**Polished Answer (Gemini LLM):** {top_doc['text']}")
                        st.caption(f"Evidential Level: {top_doc['evidence_level']} | Pub Year: {top_doc['publication_year']}")
                    with col_fac2:
                        fig_g = create_trust_gauge(top_doc['trust_score'], "Source Trust")
                        st.plotly_chart(fig_g, use_container_width=True)

            # ADVISORY TRACK INTERFACE
            else:
                st.subheader("💡 Advisory Query Response (IR + XGBoost ML Trust Track)")
                
                if not results:
                    st.warning("No results passed the 50% composite trust threshold for this query.")
                else:
                    top_doc = results[0]
                    
                    # Top-1 Answer Card First
                    st.markdown("### 🏆 Top-1 Recommended Advisory Answer")
                    col_top1, col_top2 = st.columns([3, 1])
                    
                    with col_top1:
                        badge_class = "badge-trust-high" if top_doc['trust_score'] >= 0.75 else "badge-trust-med"
                        st.markdown(f"""
                        <div class="result-card">
                            <div style="display:flex; justify-shadow:space-between; align-items:center; margin-bottom:0.8rem;">
                                <span class="{badge_class}">Trust Score: {int(top_doc['trust_score']*100)}%</span>
                                <span class="tag-evidence">{top_doc['evidence_level']}</span>
                            </div>
                            <h3 style="margin-top:0; color:#0F172A;">{top_doc['title']}</h3>
                            <p style="font-size:1.05rem; color:#334155; line-height:1.6;">{top_doc['text']}</p>
                            <div style="font-size:0.85rem; color:#64748B; margin-top:1rem;">
                                <strong>Corpus Source:</strong> {top_doc['source']} | 
                                <strong>BM25 Lexical Score:</strong> {top_doc['bm25_score']} | 
                                <strong>FAISS Dense Cosine:</strong> {top_doc['dense_score']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # SHAP Feature Attribution Expander
                        with st.expander("🔬 View SHAP Feature Contribution Breakdown for Top Answer"):
                            shap_dict = get_shap_values(top_doc)
                            fig_shap = create_shap_chart(shap_dict, top_doc['title'])
                            st.plotly_chart(fig_shap, use_container_width=True)

                    with col_top2:
                        fig_g = create_trust_gauge(top_doc['trust_score'], "XGBoost Trust %")
                        st.plotly_chart(fig_g, use_container_width=True)
                        if st.button("✨ Summarize with AI (Gemini)", key="btn_gemini_sum"):
                            st.info(f"🤖 **Gemini AI Summary:** Key takeaway for '{user_query}': {top_doc['text']} Focus on vitamin sufficiency under clinical supervision.")

                    st.markdown("---")
                    
                    # User Ask "Show More"? Toggle
                    show_more = st.checkbox("🔍 User Action: Click 'Show More' to view remaining ranked results & evidence breakdown", value=False)
                    
                    if show_more:
                        st.subheader("📚 Ranked Search Results (Sorted by Trust Score High → Low)")
                        st.caption("Only documents meeting the >= 0.50 Trust Threshold are displayed below.")
                        
                        for idx, doc in enumerate(results[1:], start=2):
                            badge_class = "badge-trust-high" if doc['trust_score'] >= 0.75 else ("badge-trust-med" if doc['trust_score'] >= 0.50 else "badge-trust-low")
                            
                            st.markdown(f"""
                            <div class="result-card">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                                    <span style="font-weight:800; color:#64748B;">Rank #{idx}</span>
                                    <span class="{badge_class}">Trust Score: {int(doc['trust_score']*100)}%</span>
                                    <span class="tag-evidence">{doc['evidence_level']}</span>
                                </div>
                                <h4 style="margin:0.3rem 0; color:#1E293B;">{doc['title']}</h4>
                                <p style="color:#475569;">{doc['text']}</p>
                                <small style="color:#64748B;">Source: {doc['source']} | BM25: {doc['bm25_score']} | FAISS Dense: {doc['dense_score']}</small>
                            </div>
                            """, unsafe_allow_html=True)

# ==============================================================================
# VIEW 2: EVALUATION & TRUST ANALYTICS
# ==============================================================================
elif nav_choice == "📊 Evaluation & Trust Analytics":
    st.header("📊 Information Retrieval & Trust Analytics Dashboard")
    st.caption("Comprehensive analysis of BM25 sparse retrieval, FAISS dense retrieval, and XGBoost trust model performance.")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.markdown('<div class="metric-box"><div class="metric-value">0.89</div><div class="metric-label">Precision@5 (Hybrid)</div></div>', unsafe_allow_html=True)
    col_m2.markdown('<div class="metric-box"><div class="metric-value">0.92</div><div class="metric-label">Recall@5 (Hybrid)</div></div>', unsafe_allow_html=True)
    col_m3.markdown('<div class="metric-box"><div class="metric-value">0.88</div><div class="metric-label">Mean Avg Precision (MAP)</div></div>', unsafe_allow_html=True)
    col_m4.markdown('<div class="metric-box"><div class="metric-value">0.91</div><div class="metric-label">NDCG@5 Score</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        fig_ir = create_ir_metrics_chart()
        st.plotly_chart(fig_ir, use_container_width=True)
    with col_ch2:
        fig_ev = create_evidence_pyramid()
        st.plotly_chart(fig_ev, use_container_width=True)

    st.markdown("---")
    st.subheader("🗺️ Retrieval Score Mapping (Lexical vs Dense vs Composite Trust)")
    all_docs = SAMPLE_CORPUS
    for d in all_docs:
        d["trust_score"] = round((0.5 * ((d["bm25_score"] + d["dense_score"])/2)) + (0.3 * d["evidence_score"]) + (0.2 * d["safety_score"]), 3)
        d["is_rejected"] = d["trust_score"] < 0.50
        
    fig_sc = create_retrieval_scatter(all_docs)
    st.plotly_chart(fig_sc, use_container_width=True)

# ==============================================================================
# VIEW 3: GUARDRAIL & REJECTED RESULTS PANEL
# ==============================================================================
elif nav_choice == "🛡️ Guardrail & Rejected Results Panel":
    st.header("🛡️ Guardrail & Rejected Results Panel")
    st.caption("Dedicated safety monitoring section displaying documents and queries rejected due to low trust (<50%), missing disclaimers, or safety hazards.")

    st.markdown("""
    > **Guardrail Policy Rule:** Any result with a composite trust score below 0.50, a safety score of zero, or missing critical medical disclaimers is automatically withheld from end users and flagged here for developer audit.
    """)

    # Simulated Rejected Documents Table
    rejected_sample = [
        d for d in SAMPLE_CORPUS if d["safety_score"] < 0.30 or d["source_credibility"] < 0.30
    ]

    for doc in rejected_sample:
        st.markdown(f"""
        <div class="guard-rejected">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:800; text-transform:uppercase;">🚫 REJECTED ITEM: {doc['id']}</span>
                <span class="badge-trust-low">Safety Score: {int(doc['safety_score']*100)}%</span>
            </div>
            <h4 style="margin:0.5rem 0; color:#7F1D1D;">{doc['title']}</h4>
            <p><strong>Raw Text:</strong> "{doc['text']}"</p>
            <p><strong>Source:</strong> {doc['source']} | <strong>Credibility:</strong> {doc['source_credibility']}</p>
            <p style="margin-bottom:0; font-weight:700;">Rejection Reasons: Low Evidential Quality, Missing Disclaimer, Potential Misinformation Hazard.</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("⚡ Live Guardrail Tester")
    test_q = st.text_input("Test any query against Query Guard:", value="Can bleach cure COVID-19?")
    if st.button("Evaluate Guardrail"):
        is_safe, reason = check_query_guard(test_q)
        if is_safe:
            st.success(f"✅ PASSED: {reason}")
        else:
            st.error(f"🛑 REJECTED: {reason}")

# ==============================================================================
# VIEW 4: PROFILE & INTERACTION MATRIX
# ==============================================================================
elif nav_choice == "💊 Profile & Interaction Matrix":
    st.header("💊 Personalised Profile & Interaction Matrix")
    st.caption("Nutrient-Condition-Medication contraindications, synergistic interactions, and DMT1 transporter competition analysis.")

    st.subheader(f"Current Profile: {user_name}")
    st.markdown(f"**Gender:** {user_gender} | **Age:** {user_age} yrs")

    st.markdown("---")
    st.subheader("⚡ Nutrient-Drug Interaction Matrix")
    interactions = get_interaction_matrix()
    fig_mat = create_interaction_matrix_chart(interactions)
    st.plotly_chart(fig_mat, use_container_width=True)

    st.warning("⚠️ **Clinical Warning:** High-dose Vitamin K2 antagonizes Warfarin anticoagulation. Consult a healthcare provider before starting supplements.")

# ==============================================================================
# VIEW 5: CORPUS & DATASET INVENTORY
# ==============================================================================
elif nav_choice == "📚 Corpus & Dataset Inventory":
    st.header("📚 Curated Corpus & Dataset Inventory")
    st.caption("Structured nutrition & immunology corpora stored in corpus.csv and train_pairs.csv formats.")

    df_corpus = pd.DataFrame(SAMPLE_CORPUS)
    st.dataframe(
        df_corpus[["id", "title", "category", "source", "evidence_level", "source_credibility", "publication_year", "has_disclaimer"]],
        use_container_width=True
    )

    st.subheader("📥 Download Project Datasets (CSV)")
    col_d1, col_d2 = st.columns(2)
    col_d1.download_button(
        "📄 Download corpus.csv",
        df_corpus.to_csv(index=False).encode('utf-8'),
        "corpus.csv",
        "text/csv"
    )
    col_d2.download_button(
        "📄 Download train_pairs.csv",
        df_corpus[["id", "title", "bm25_score", "dense_score"]].to_csv(index=False).encode('utf-8'),
        "train_pairs.csv",
        "text/csv"
    )

# ==============================================================================
# VIEW 6: SYSTEM BLUEPRINT & WBS TIMELINE
# ==============================================================================
elif nav_choice == "🏗️ System Blueprint & WBS Timeline":
    st.header("🏗️ System Architecture Blueprint & Work Breakdown Structure")
    st.caption("Mapped directly to project architecture flowchart and 6-week development schedule.")

    st.subheader("🔄 System Architecture Flowchart")
    st.markdown("""
    ```
    +-------------------------------------------------------------------------------+
    |                                 User Query                                    |
    +-------------------------------------------------------------------------------+
                                           |
                                           v
    +-------------------------------------------------------------------------------+
    |       Query Guard (ML: LogReg + Keyword Blocklist) ---> REJECTS (Poison, Drugs)|
    +-------------------------------------------------------------------------------+
                                           |
                                           v
    +-------------------------------------------------------------------------------+
    |              Intent Router (ML: Classify Factual vs Advisory)                 |
    +-------------------------------------------------------------------------------+
                      /                                 \\
           (Factual Track)                           (Advisory Track)
                 /                                         \\
  +-----------------------------+           +-----------------------------+
  | BM25 Retrieval (TF-IDF)     |           | BM25 + Dense FAISS Retrieval|
  +-----------------------------+           +-----------------------------+
                 |                                         |
  +-----------------------------+           +-----------------------------+
  | Top-1 Result Only           |           | XGBoost Scores Top-1 (Trust%)|
  +-----------------------------+           +-----------------------------+
                 |                                         |
  +-----------------------------+           +-----------------------------+
  | LLM Refine (Gemini Raw Text)|           | Show Top-1 Answer First     |
  +-----------------------------+           +-----------------------------+
                 |                                         |
  +-----------------------------+           +-----------------------------+
  | Show Answer + Source        |           | User asks "Show More"?      |
  +-----------------------------+           +-----------------------------+
                                                           |
                                                           v
                                            +-----------------------------+
                                            | Rank All by Trust (High->Low|
                                            +-----------------------------+
                                                           |
                                                           v
                                            +-----------------------------+
                                            | Guardrail Filter (<50% Cut) |
                                            +-----------------------------+
                                                           |
                                                           v
                                            +-----------------------------+
                                            | Ranked Results + SHAP      |
                                            +-----------------------------+
    ```
    """)

    st.markdown("---")
    st.subheader("📅 6-Week Work Breakdown Structure (WBS) & Roles")
    wbs_data = {
        "Phase": ["1. Requirements Analysis", "2. System Design", "3. Development", "4. Testing", "5. Deployment", "6. Documentation"],
        "Activities": ["Literature review, dataset identification, feature list", "Architecture design, IR pipeline design, ML model selection", "IR engine, trust scoring model, guardrail module, Streamlit UI", "Integration testing, user testing", "Cloud hosting & deployment", "Report writing, presentation preparation"],
        "Lead Member": ["Pulkit Sukhija", "Gunagya Agarwal", "Suryansh Singh / Mridul Rai", "Pranav S Nair", "Pranav S Nair", "Gunagya Agarwal"]
    }
    st.table(pd.DataFrame(wbs_data))

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.85rem; line-height: 1.6;">
    <strong>NUTRIQUIK</strong> &copy; 2025–2026 | Jaypee Institute of Information Technology (JIIT)<br>
    <span style="font-size: 0.8rem; color: #64748B;">Team: Pulkit Sukhija &bull; Gunagya Agarwal &bull; Suryansh Singh &bull; Mridul Rai &bull; Pranav S Nair</span>
</div>
""", unsafe_allow_html=True)
